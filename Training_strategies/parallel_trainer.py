import sys
import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Process, Manager
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
import argparse
import threading
import time
from collections import deque
import copy

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)

class SharedReplayBuffer:
    """Thread-safe replay buffer for parallel training."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        with self.lock:
            self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample batch from buffer."""
        with self.lock:
            if len(self.buffer) < batch_size:
                return None
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
        return batch
    
    def __len__(self):
        with self.lock:
            return len(self.buffer)

class ParallelDQNTradingTrainer:
    """
    Parallel DQN Trading Trainer with multiple workers and asynchronous policy updates.
    
    This implementation uses:
    1. Multiple worker processes for parallel episode execution
    2. Shared replay buffer for experience collection
    3. Asynchronous policy updates on the main process
    4. Periodic synchronization of worker networks
    """
    
    def __init__(self, 
                 symbol: str,
                 start_date: str,
                 end_date: str,
                 target_profit: float = 1.005,
                 stop_loss: float = 0.99,
                 num_epochs: int = 3000,
                 num_workers: int = 4,
                 save_folder: str = None,
                 model_path: str = None,
                 norm_params_path: str = None,
                 device: str = None,
                 replay_buffer_size: int = 100000,
                 batch_size: int = 32,
                 update_frequency: int = 4,
                 sync_frequency: int = 10):
        """
        Initialize the Parallel DQN Trading Trainer.
        
        Args:
            num_workers: Number of parallel worker processes
            replay_buffer_size: Size of shared replay buffer
            batch_size: Batch size for policy updates
            update_frequency: Frequency of policy updates (every N steps)
            sync_frequency: Frequency of network synchronization (every N episodes)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.target_profit = target_profit
        self.stop_loss = stop_loss
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        
        # Parallel training parameters
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.sync_frequency = sync_frequency
        
        # Set up paths
        self.save_folder = save_folder or f"./trained_models/{symbol.lower()}_{start_date.replace('-', '_')}"
        self.model_path = model_path
        self.norm_params_path = norm_params_path or f"./json_files/{symbol.lower()}_{start_date.replace('-', '_')}_norm_params.json"
        self.weights_folder = os.path.join(self.save_folder, "model_weights")
        self.stats_csv = os.path.join(self.save_folder, f"model_stats/{symbol}_{start_date}_stats.csv")
        
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create save directory
        os.makedirs(self.save_folder, exist_ok=True)
        os.makedirs(os.path.dirname(self.norm_params_path), exist_ok=True)
        os.makedirs(self.weights_folder, exist_ok=True)
        
        # Initialize data containers
        self.df = None
        self.df_normalized = None
        self.norm_params = None
        self.target_buy = None
        self.target_sell = None
        
        # Initialize models and training components
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.shared_replay_buffer = SharedReplayBuffer(replay_buffer_size)
        
        # Multiprocessing components
        self.manager = Manager()
        self.worker_queues = []
        self.result_queue = Queue()
        self.workers = []
        self.episode_counter = self.manager.Value('i', 0)
        self.global_step = self.manager.Value('i', 0)
        
        # Training statistics
        self.training_stats = []
        
    def load_data(self, additional_csv_path: str = None) -> pd.DataFrame:
        """Load and prepare stock data (same as original)."""
        print(f"Loading data for {self.symbol} from {self.start_date} to {self.end_date}")
        
        from utils_data import UpstoxStockDataFetcher, getTechnicalIndicators
        from utils_data import generateTargetDataBuySide, generateTargetDataSellSide
        from utils_data import normalize_dataframe_with_mean_std
        
        fetcher = UpstoxStockDataFetcher()
        self.df = fetcher.get_stock_data(self.symbol, self.start_date, self.end_date)
        
        if additional_csv_path and os.path.exists(additional_csv_path):
            df_additional = pd.read_csv(additional_csv_path)
            df_additional = df_additional.drop_duplicates()
            df_additional['time'] = df_additional['time'].astype(str) + "+05:30"
            df_additional['time'] = pd.to_datetime(df_additional['time'])
            self.df = pd.concat([self.df, df_additional], ignore_index=True)
            print(f"Added additional data from {additional_csv_path}")
        
        print(f"Loaded {len(self.df)} data points")
        return self.df
    
    def prepare_features_and_targets(self):
        """Generate technical indicators and target data (same as original)."""
        print("Generating technical indicators...")
        from utils_data import getTechnicalIndicators, generateTargetDataBuySide, generateTargetDataSellSide
        
        self.df = getTechnicalIndicators(self.df)
        
        print("Generating target data...")
        self.target_buy = generateTargetDataBuySide(
            self.df, 
            self.target_profit, 
            self.stop_loss
        )
        
        sell_target = 1 / self.target_profit
        sell_stop = 1 / self.stop_loss
        
        self.target_sell = generateTargetDataSellSide(
            self.df,
            sell_target,
            sell_stop
        )
        
        self._print_target_statistics()
        
    def _print_target_statistics(self):
        """Print statistics about target data."""
        print("\nBuy Side Statistics:")
        buy_stats = self.target_buy['action'].value_counts()
        for action, count in buy_stats.items():
            print(f"  {action}: {count}")
        
        print("\nSell Side Statistics:")
        sell_stats = self.target_sell['action'].value_counts()
        for action, count in sell_stats.items():
            print(f"  {action}: {count}")
    
    def normalize_data(self):
        """Normalize the dataframe and save normalization parameters."""
        print("Normalizing data...")
        from utils_data import normalize_dataframe_with_mean_std
        
        self.df_normalized, self.norm_params = normalize_dataframe_with_mean_std(self.df)
        
        with open(self.norm_params_path, "w") as f:
            json.dump(self.norm_params, f)
        print(f"Normalization parameters saved to {self.norm_params_path}")
        
    def initialize_models(self):
        """Initialize DQN models and optimizer."""
        print("Initializing models...")
        from Models.DQN import DQN
        
        self.policy_net = DQN(16, 3)
        self.target_net = DQN(16, 3)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading pre-trained model from {self.model_path}")
            self.policy_net.load_state_dict(torch.load(self.model_path))
        
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        print(f"Models initialized on device: {self.device}")
    
    def get_state(self, df: pd.DataFrame, current_step: int) -> np.ndarray:
        """Extract state features from dataframe at given step."""
        row = df.iloc[current_step]
        state = np.array([
            row['time'],
            row['open'],
            row['high'],
            row['low'],
            row['close'],
            row['volume'],
            row['MA50'],
            row['RSI'],
            row['MACD'],
            row['BB_upper'],
            row['BB_lower'],
            row['ADX'],
            row['CCI'],
            row['ATR'],
            row['ROC'],
            row['OBV']
        ], dtype=np.float32)
        return state
    
    def calculate_optimized_scalping_reward(self, 
                                         delay: float, 
                                         action_type: str, 
                                         success_base_reward: float = 1500,
                                         failure_base_penalty: float = 1000, 
                                         min_delay_threshold: float = 60,
                                         max_reward: float = 2500, 
                                         decay_rate: float = 0.3,
                                         opportunity_cost_factor: float = 0.2,
                                         missed_opp_multiplier: float = 2.0,
                                         consecutive_successes: int = 0,
                                         consecutive_success_bonus: float = 0.15) -> float:
        """Comprehensive reward function optimized for scalping."""
        delay = delay / 60  # Convert to minutes
        
        if action_type == 'success':
            if delay <= min_delay_threshold:
                base_reward = max_reward - (max_reward - success_base_reward) * (delay / min_delay_threshold)
            else:
                base_reward = success_base_reward
            
            reward = base_reward * np.exp(-decay_rate * delay)
            opportunity_cost = opportunity_cost_factor * delay * success_base_reward
            opportunity_cost = min(opportunity_cost, reward * 0.8)
            reward = reward - opportunity_cost
            
            if consecutive_successes > 0:
                sequential_bonus = reward * (consecutive_success_bonus * consecutive_successes)
                reward += sequential_bonus
            
            return reward
        
        elif action_type == 'failure':
            penalty = -failure_base_penalty * np.exp(-decay_rate * delay)
            opportunity_cost = opportunity_cost_factor * delay * failure_base_penalty
            penalty = penalty - opportunity_cost
            return penalty
        
        elif action_type == 'missed_opportunity':
            missed_penalty = -failure_base_penalty * missed_opp_multiplier * np.exp(-decay_rate * delay)
            return missed_penalty
        
        elif action_type == 'no_action':
            return 100
        
        return 0
    
    def worker_process(self, worker_id: int, shared_state_dict: dict, episode_queue: Queue):
        """
        Worker process for parallel episode execution.
        
        Args:
            worker_id: Unique identifier for the worker
            shared_state_dict: Shared network state dictionary
            episode_queue: Queue to receive episode assignments
        """
        print(f"Worker {worker_id} started")
        
        # Initialize worker's local network
        from Models.DQN import DQN, DQNAgent
        from trading_environment import StockTradingEnv
        
        local_policy_net = DQN(16, 3)
        local_policy_net.load_state_dict(shared_state_dict)
        local_policy_net.eval()  # Workers only do inference
        
        env = StockTradingEnv(self.df_normalized)
        agent = DQNAgent(env, local_policy_net, local_policy_net)
        agent.epsilon = max(0.1, 1.0 - worker_id * 0.1)  # Different exploration rates per worker
        
        while True:
            try:
                # Get episode assignment
                episode_data = episode_queue.get(timeout=1)
                if episode_data is None:  # Shutdown signal
                    break
                
                episode_num = episode_data['episode']
                network_state = episode_data.get('network_state')
                
                # Update local network if new state provided
                if network_state:
                    local_policy_net.load_state_dict(network_state)
                
                # Run episode
                episode_stats = self.run_worker_episode(agent, worker_id, episode_num)
                
                # Send results back
                self.result_queue.put({
                    'worker_id': worker_id,
                    'episode': episode_num,
                    'stats': episode_stats
                })
                
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                break
        
        print(f"Worker {worker_id} terminated")
    
    def run_worker_episode(self, agent, worker_id: int, episode: int) -> Dict[str, Any]:
        """Run a single episode in a worker process."""
        total_reward = 0
        number_trans = 0
        wins = 0
        lose = 0
        defeat = 0
        consecutive_success = 0
        
        step = 0
        experiences = []  # Collect experiences for batch upload to replay buffer
        
        while step < len(self.df_normalized):
            state = self.get_state(self.df_normalized, step)
            action = agent.select_action(state)
            done = False
            reward = 0
            
            if action == 1:  # BUY
                next_state = self.target_buy.iloc[step]
                next_state_index = next_state["next_state_index"]
                next_state2 = self.df_normalized.iloc[next_state_index].copy()
                
                if next_state['action'] == "Target Hit":
                    wins += 1
                    consecutive_success += 1
                    reward = self.calculate_optimized_scalping_reward(
                        delay=self.target_buy.iloc[step]['delay'],
                        action_type="success",
                        consecutive_successes=consecutive_success
                    )
                elif next_state['action'] == "Stop Loss Hit":
                    defeat += 1
                    consecutive_success = 0
                    reward = self.calculate_optimized_scalping_reward(
                        delay=self.target_buy.iloc[step]['delay'],
                        action_type="failure"
                    )
                elif next_state['action'] == "End of Day":
                    lose += 1
                    consecutive_success = 0
                    done = True
                    reward = self.calculate_optimized_scalping_reward(
                        delay=self.target_buy.iloc[step]['delay'],
                        action_type="failure"
                    )
                
                next_state2 = np.array(next_state2.values, dtype=np.float32)
                experiences.append((state, action, float(reward), next_state2, done))
                number_trans += 1
                step = next_state_index + 1
                
            elif action == 2:  # SELL
                next_state = self.target_sell.iloc[step]
                next_state_index = next_state["next_state_index"]
                next_state2 = self.df_normalized.iloc[next_state_index].copy()
                
                if next_state['action'] == "Target Hit":
                    wins += 1
                    consecutive_success += 1
                    reward = self.calculate_optimized_scalping_reward(
                        delay=self.target_sell.iloc[step]['delay'],
                        action_type="success",
                        consecutive_successes=consecutive_success
                    )
                elif next_state['action'] == "Stop Loss Hit":
                    consecutive_success = 0
                    defeat += 1
                    reward = self.calculate_optimized_scalping_reward(
                        delay=self.target_sell.iloc[step]['delay'],
                        action_type="failure"
                    )
                elif next_state['action'] == "End of Day":
                    consecutive_success = 0
                    lose += 1
                    done = True
                    reward = self.calculate_optimized_scalping_reward(
                        delay=self.target_sell.iloc[step]['delay'],
                        action_type="failure"
                    )
                
                next_state2 = np.array(next_state2.values, dtype=np.float32)
                experiences.append((state, action, float(reward), next_state2, done))
                number_trans += 1
                step = next_state_index + 1
                
            elif action == 0:  # HOLD
                buy_side = self.target_buy.iloc[step]
                sell_side = self.target_sell.iloc[step]
                
                if buy_side['action'] == "Target Hit":
                    reward = self.calculate_optimized_scalping_reward(
                        delay=self.target_buy.iloc[step]['delay'],
                        action_type="missed_opportunity"
                    )
                elif sell_side['action'] == "Target Hit":
                    reward = self.calculate_optimized_scalping_reward(
                        delay=self.target_sell.iloc[step]['delay'],
                        action_type="missed_opportunity"
                    )
                else:
                    reward = 100
                
                if step + 1 < len(self.df_normalized):
                    next_state = self.get_state(self.df_normalized, step + 1)
                else:
                    done = True
                    next_state = self.get_state(self.df_normalized, -1)
                
                experiences.append((state, action, float(reward), next_state, done))
                step = step + 1
            
            total_reward += reward
        
        # Batch upload all experiences to shared replay buffer
        for exp in experiences:
            self.shared_replay_buffer.push(*exp)
        
        return {
            'episode': episode,
            'worker_id': worker_id,
            'total_reward': total_reward,
            'number_trans': number_trans,
            'wins': wins,
            'lose': lose,
            'defeat': defeat,
            'win_rate': wins / max(number_trans, 1) * 100,
            'experiences_collected': len(experiences)
        }
    
    def update_policy_async(self):
        """Asynchronous policy update using shared replay buffer."""
        if len(self.shared_replay_buffer) < self.batch_size:
            return False
        
        # Sample batch from replay buffer
        batch = self.shared_replay_buffer.sample(self.batch_size)
        if batch is None:
            return False
        
        # Convert batch to tensors
        states = torch.FloatTensor([exp[0] for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp[1] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp[3] for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp[4] for exp in batch]).to(self.device)
        
        # Compute Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        # Compute loss and update
        loss = torch.nn.functional.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return True
    
    def train_parallel(self, start_episode: int = 0, save_frequency: int = 5):
        """
        Main parallel training loop.
        
        Args:
            start_episode: Episode to start training from
            save_frequency: Frequency of model saving
        """
        print(f"Starting parallel training with {self.num_workers} workers")
        print(f"Total episodes: {self.num_epochs}")
        
        # Initialize episode queue for workers
        episode_queue = Queue()
        
        # Start worker processes
        for worker_id in range(self.num_workers):
            worker = Process(
                target=self.worker_process,
                args=(worker_id, self.policy_net.state_dict(), episode_queue)
            )
            worker.start()
            self.workers.append(worker)
        
        # Training loop
        episodes_completed = start_episode
        episodes_assigned = start_episode
        update_counter = 0
        
        # Create progress bar
        pbar = tqdm(total=self.num_epochs, initial=start_episode, desc="Training Progress")
        
        try:
            # Assign initial episodes to workers
            for _ in range(min(self.num_workers * 2, self.num_epochs - episodes_assigned)):
                episode_queue.put({'episode': episodes_assigned})
                episodes_assigned += 1
            
            while episodes_completed < self.num_epochs:
                # Process completed episodes
                try:
                    result = self.result_queue.get(timeout=1)
                    episodes_completed += 1
                    pbar.update(1)
                    
                    # Store statistics
                    episode_stats = result['stats']
                    self.training_stats.append(episode_stats)
                    
                    # Save episode statistics
                    episode_df = pd.DataFrame([episode_stats])
                    if not os.path.exists(self.stats_csv):
                        episode_df.to_csv(self.stats_csv, index=False)
                    else:
                        episode_df.to_csv(self.stats_csv, mode='a', index=False, header=False)
                    
                    # Assign new episode if available
                    if episodes_assigned < self.num_epochs:
                        # Periodically send updated network weights to workers
                        network_state = None
                        if episodes_assigned % self.sync_frequency == 0:
                            network_state = self.policy_net.state_dict()
                        
                        episode_queue.put({
                            'episode': episodes_assigned,
                            'network_state': network_state
                        })
                        episodes_assigned += 1
                    
                    # Perform asynchronous policy updates
                    if update_counter % self.update_frequency == 0:
                        if self.update_policy_async():
                            update_counter = 0
                    update_counter += 1
                    
                    # Update target network and save model periodically
                    if episodes_completed % save_frequency == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                        
                        # Save model
                        model_save_path = os.path.join(
                            self.weights_folder,
                            f'{self.symbol.lower()}_{self.start_date.replace("-", "_")}_{episodes_completed}.pth'
                        )
                        torch.save(self.policy_net.state_dict(), model_save_path)
                        
                        # Print statistics for recent episodes
                        recent_stats = self.training_stats[-save_frequency:]
                        avg_reward = np.mean([s['total_reward'] for s in recent_stats])
                        avg_win_rate = np.mean([s['win_rate'] for s in recent_stats])
                        avg_trans = np.mean([s['number_trans'] for s in recent_stats])
                        
                        print(f"\nEpisode {episodes_completed}/{self.num_epochs}")
                        print(f"Avg Reward (last {save_frequency}): {avg_reward:.2f}")
                        print(f"Avg Win Rate (last {save_frequency}): {avg_win_rate:.2f}%")
                        print(f"Avg Transactions (last {save_frequency}): {avg_trans:.1f}")
                        print(f"Replay Buffer Size: {len(self.shared_replay_buffer)}")
                        print(f"Model saved to: {model_save_path}")
                
                except:
                    # Continue if no results available
                    time.sleep(0.01)
                    continue
        
        finally:
            pbar.close()
            
            # Shutdown workers
            print("Shutting down workers...")
            for _ in range(self.num_workers):
                episode_queue.put(None)  # Shutdown signal
            
            for worker in self.workers:
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.terminate()
    
    def run_full_training_pipeline(self, additional_csv_path: str = None, start_episode: int = 0):
        """Run the complete parallel training pipeline."""
        print(f"Starting parallel training pipeline for {self.symbol}")
        print(f"Training parameters:")
        print(f"  Target profit: {self.target_profit}")
        print(f"  Stop loss: {self.stop_loss}")
        print(f"  Number of epochs: {self.num_epochs}")
        print(f"  Number of workers: {self.num_workers}")
        print(f"  Device: {self.device}")
        print(f"  Replay buffer size: {self.replay_buffer_size}")
        print(f"  Batch size: {self.batch_size}")
        
        # Load and prepare data
        self.load_data(additional_csv_path)
        self.prepare_features_and_targets()
        self.normalize_data()
        
        # Initialize models
        self.initialize_models()
        
        # Start parallel training
        self.train_parallel(start_episode)
        
        print("Parallel training completed successfully!")

# Example usage
if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Train DQN Trading Agent in Parallel")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--num_epochs", type=int, default=3000, help="Number of training episodes")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--target_profit", type=float, default=1.005, help="Target profit multiplier")
    parser.add_argument("--stop_loss", type=float, default=0.99, help="Stop loss multiplier")
    parser.add_argument("--save_folder", type=str, default=None, help="Folder to save trained models")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pre-trained model")
    parser.add_argument("--norm_params_path", type=str, default=None, help="Path to normalization params")
    parser.add_argument("--additional_csv", type=str, default=None, help="Path to additional CSV data")
    parser.add_argument("--start_episode", type=int, default=0, help="Episode to resume from")
    parser.add_argument("--replay_buffer_size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for updates")
    parser.add_argument("--update_frequency", type=int, default=4, help="Policy update frequency")
    parser.add_argument("--sync_frequency", type=int, default=10, help="Network sync frequency")
    
    args = parser.parse_args()

    trainer = ParallelDQNTradingTrainer(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        target_profit=args.target_profit,
        stop_loss=args.stop_loss,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        save_folder=args.save_folder,
        model_path=args.model_path,
        norm_params_path=args.norm_params_path,
        replay_buffer_size=args.replay_buffer_size,
        batch_size=args.batch_size,
        update_frequency=args.update_frequency,
        sync_frequency=args.sync_frequency
    )

    trainer.run_full_training_pipeline(
        additional_csv_path=args.additional_csv,
        start_episode=args.start_episode
    )

# Example command to run with parallel training:
# python parallel_trainer.py \
#     --symbol SUZLON \
#     --start_date 2025-06-15 \
#     --end_date 2025-06-19 \
#     --num_epochs 3000 \
#     --num_workers 8 \
#     --target_profit 1.005 \
#     --stop_loss 0.99 \
#     --replay_buffer_size 200000 \
#     --batch_size 64 \
#     --update_frequency 4 \
#     --sync_frequency 10