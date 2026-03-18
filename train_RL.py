"""
Train DDPG agent using the pretrained LNN as simulator.
Usage: python scripts/train_rl.py --lnn checkpoints/lnn_best.pth
"""
import argparse
import torch
import numpy as np
import sys
sys.path.append('.')
from models.lnn import PhysicsInformedLNN
from models.ddpg import DDPGAgent, ReplayBuffer
from utils.data_loader import load_and_preprocess

# Define environment simulator using LNN
class LENSSimulator:
    def __init__(self, lnn_model, scaler_X, scaler_y, target_height=0.3):
        self.lnn = lnn_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.target_height = target_height
        self.device = next(lnn_model.parameters()).device
        self.reset()

    def reset(self):
        # Random initial parameters within bounds
        self.P = np.random.uniform(1.0, 2.0)
        self.V = np.random.uniform(4.0, 12.0)
        self.F = np.random.uniform(6.0, 12.0)
        self.eta = 35.0  # fixed
        self.material_idx = np.random.choice(3)  # 0:Ni60A,1:Stellite6,2:In625
        self._update_state()
        return self.state

    def _update_state(self):
        # Prepare input for LNN
        material_onehot = np.zeros(3)
        material_onehot[self.material_idx] = 1
        X_raw = np.array([[self.P, self.V, self.F, self.eta] + material_onehot.tolist()])
        X_scaled = self.scaler_X.transform(X_raw)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            y_pred, T_pred, sigma_pred = self.lnn(X_tensor)
        y_orig = self.scaler_y.inverse_transform(y_pred.cpu().numpy())
        self.D, self.HV, self.Ra, self.CUI = y_orig[0]
        self.T_peak = T_pred.item() * 2000  # denormalize approx
        self.P_crack = max(0, min(1, (self.D - 15) / 20))  # heuristic crack prob
        self.V_dot = 0.8 * self.F * 0.5  # eta_c * F * A_bead
        self.h_layer = 0.3 + 0.1 * np.random.randn()  # simplified
        # State: [T_peak, f_osc, P_crack, V_dot] normalized
        self.state = np.array([
            self.T_peak / 2000,
            0.5,  # f_osc placeholder
            self.P_crack,
            self.V_dot / 100
        ])

    def step(self, action):
        # action: [ΔP, ΔV, ΔF] in [-1,1]
        delta_P, delta_V, delta_F = action
        self.P *= (1 + delta_P * 0.1)
        self.V *= (1 + delta_V * 0.1)
        self.F *= (1 + delta_F * 0.1)
        # Clip to bounds
        self.P = np.clip(self.P, 1.0, 2.0)
        self.V = np.clip(self.V, 4.0, 12.0)
        self.F = np.clip(self.F, 6.0, 12.0)
        self._update_state()
        # Reward
        delta_h = self.h_layer - self.target_height
        E_inst = self.P * 0.1  # kW * dt
        reward = -0.5 * (delta_h**2) - 0.3 * self.P_crack - 0.2 * E_inst
        return self.state, reward, False, {}

def train_rl(lnn_checkpoint, config):
    # Load LNN and scalers
    _, _, _, _, _, _, scaler_X, scaler_y, _ = load_and_preprocess('data/experimental_data.csv')
    input_dim = 7  # 4 cont + 3 one-hot
    lnn = PhysicsInformedLNN(input_dim)
    lnn.load_state_dict(torch.load(lnn_checkpoint))
    lnn.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lnn.to(device)

    env = LENSSimulator(lnn, scaler_X, scaler_y)
    agent = DDPGAgent(state_dim=4, action_dim=3, lr=config['lr'], gamma=config['gamma'], tau=config['tau'])
    replay_buffer = ReplayBuffer(capacity=int(config['buffer_size']))

    # Ornstein-Uhlenbeck noise
    class OUNoise:
        def __init__(self, action_dim, theta=0.15, sigma=0.2, dt=0.01):
            self.theta = theta
            self.sigma = sigma
            self.dt = dt
            self.reset()
        def reset(self):
            self.state = np.zeros(self.action_dim)
        def sample(self):
            self.state += self.theta * (-self.state) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
            return self.state

    noise = OUNoise(3)

    episode_rewards = []
    for episode in range(config['episodes']):
        state = env.reset()
        noise.reset()
        episode_reward = 0
        for step in range(config['max_steps']):
            action = agent.select_action(state, noise_scale=0.0)
            # Add exploration noise
            action = np.clip(action + noise.sample(), -1, 1)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > config['batch_size']:
                agent.update(replay_buffer, config['batch_size'])
            state = next_state
            episode_reward += reward
            if done:
                break
        episode_rewards.append(episode_reward)
        if (episode+1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f'Episode {episode+1}, avg reward: {avg_reward:.4f}')

    agent.save('checkpoints/ddpg_final.pth')
    print('Training completed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lnn', type=str, required=True, help='Path to trained LNN checkpoint')
    parser.add_argument('--config', type=str, default='config/rl_config.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    train_rl(args.lnn, config)