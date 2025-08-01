import sys
import os
import json
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)

class PPOTradingTrainer:
    """
    A comprehensive trainer class for PPO-based stock trading models.
    
    This class handles the entire training pipeline including:
    - Data fetching and preprocessing
    - Target generation for buy/sell strategies
    - Model training with configurable parameters
    - Model saving and checkpointing
    """
    
    def __init__(self, 
                 symbol: str,
                 start_date: str,
                 end_date: str,
                 target_profit: float = 1.005,
                 stop_loss: float = 0.99,
                 num_epochs: int = 3000,
                 save_folder: str = None,
                 actor_model_path: str = None,
                 critic_model_path: str = None,
                 norm_params_path: str = None,
                 device: str = None,
                 update_frequency: int = 2048):  # PPO update frequency
        """
        Initialize the PPO Trading Trainer.
        
        Args:
            symbol: Stock symbol to trade (e.g., 'SUZLON')
            start_date: Start date for data fetching (YYYY-MM-DD)
            end_date: End date for data fetching (YYYY-MM-DD)
            target_profit: Target profit multiplier for buy side (default: 1.005 = 0.5% profit)
            stop_loss: Stop loss multiplier for buy side (default: 0.99 = 1% loss)
            num_epochs: Number of training episodes
            save_folder: Folder to save trained models
            actor_model_path: Path to pre-trained actor model (optional)
            critic_model_path: Path to pre-trained critic model (optional)
            norm_params_path: Path to save normalization parameters
            device: Device to use for training ('cuda' or 'cpu')
            update_frequency: How often to update PPO policy (in steps)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.target_profit = target_profit
        self.stop_loss = stop_loss
        self.num_epochs = num_epochs
        self.update_frequency = update_frequency
        
        # Set up paths
        self.save_folder = save_folder or f"./trained_models/{symbol.lower()}_{start_date.replace('-', '_')}"
        self.actor_model_path = actor_model_path
        self.critic_model_path = critic_model_path
        self.norm_params_path = norm_params_path or f"./json_files/{symbol.lower()}_{start_date.replace('-', '_')}_norm_params.json"
        self.weights_folder = os.path.join(self.save_folder, "model_weights")
        self.stats_csv = os.path.join(self.save_folder, f"model_stats/{symbol}_{start_date}_stats.csv")
        
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training will use device: {self.device}")
        
        # Create save directory
        os.makedirs(self.save_folder, exist_ok=True)
        os.makedirs(os.path.dirname(self.norm_params_path), exist_ok=True)
        os.makedirs(self.weights_folder, exist_ok=True)
        os.makedirs(os.path.dirname(self.stats_csv), exist_ok=True)
        
        # Initialize data containers
        self.df = None
        self.df_normalized = None
        self.norm_params = None
        self.target_buy = None
        self.target_sell = None
        
        # Initialize models
        self.agent = None
        self.env = None
        
        # PPO specific counters
        self.step_count = 0
        
    def load_data(self, additional_csv_path: str = None) -> pd.DataFrame:
        """
        Load and prepare stock data.
        
        Args:
            additional_csv_path: Path to additional CSV data to concatenate
            
        Returns:
            Combined dataframe with stock data
        """
        print(f"Loading data for {self.symbol} from {self.start_date} to {self.end_date}")
        
        # Import required modules (these should be available in your environment)
        from utils_data import UpstoxStockDataFetcher, getTechnicalIndicators
        from utils_data import generateTargetDataBuySide, generateTargetDataSellSide
        from utils_data import normalize_dataframe_with_mean_std
        
        # Fetch main data
        fetcher = UpstoxStockDataFetcher()
        self.df = fetcher.get_stock_data(self.symbol, self.start_date, self.end_date)
        
        # Add additional CSV data if provided
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
        """
        Generate technical indicators and target data for training.
        """
        print("Generating technical indicators...")
        from utils_data import getTechnicalIndicators, generateTargetDataBuySide, generateTargetDataSellSide
        
        # Add technical indicators
        self.df = getTechnicalIndicators(self.df)
        
        # Generate target data
        print("Generating target data...")
        self.target_buy = generateTargetDataBuySide(
            self.df, 
            self.target_profit, 
            self.stop_loss
        )
        
        # For sell side, invert the profit/loss ratios
        sell_target = 1 / self.target_profit  # If buy target is 1.005, sell target is ~0.995
        sell_stop = 1 / self.stop_loss        # If buy stop is 0.99, sell stop is ~1.01
        
        self.target_sell = generateTargetDataSellSide(
            self.df,
            sell_target,
            sell_stop
        )
        
        # Print target statistics
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
        """
        Normalize the dataframe and save normalization parameters.
        """
        print("Normalizing data...")
        from utils_data import normalize_dataframe_with_mean_std
        
        self.df_normalized, self.norm_params = normalize_dataframe_with_mean_std(self.df)
        
        # Save normalization parameters
        with open(self.norm_params_path, "w") as f:
            json.dump(self.norm_params, f)
        print(f"Normalization parameters saved to {self.norm_params_path}")
        
    def initialize_models(self):
        """
        Initialize PPO models and training environment.
        """
        print("Initializing PPO models...")
        from Models.PPO import PPOAgent  # Import your PPO implementation
        from trading_environment import StockTradingEnv
        
        # Initialize environment
        self.env = StockTradingEnv(self.df_normalized)
        
        # Initialize PPO agent
        self.agent = PPOAgent(
            env=self.env,
            input_dim=16,  # 16 features
            output_dim=3,  # 3 actions (hold, buy, sell)
            lr=3e-4,
            gamma=0.99,
            clip_ratio=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            update_epochs=10,
            batch_size=64
        )
        
        # Load pre-trained models if provided
        if self.actor_model_path and self.critic_model_path:
            if os.path.exists(self.actor_model_path) and os.path.exists(self.critic_model_path):
                print(f"Loading pre-trained models from {self.actor_model_path} and {self.critic_model_path}")
                self.agent.load_models(self.actor_model_path, self.critic_model_path)
        
        print(f"PPO models initialized on device: {self.device}")
    
    def get_state(self, df: pd.DataFrame, current_step: int) -> np.ndarray:
        """
        Extract state features from dataframe at given step.
        
        Args:
            df: Input dataframe
            current_step: Current step index
            
        Returns:
            State vector as numpy array
        """
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
        """
        Comprehensive reward function optimized for scalping.
        """
        delay = delay / 60  # Convert to minutes
        
        if action_type == 'success':
            # Delay-dependent base reward scaling
            if delay <= min_delay_threshold:
                base_reward = max_reward - (max_reward - success_base_reward) * (delay / min_delay_threshold)
            else:
                base_reward = success_base_reward
            
            # Apply exponential decay
            reward = base_reward * np.exp(-decay_rate * delay)
            
            # Apply opportunity cost
            opportunity_cost = opportunity_cost_factor * delay * success_base_reward
            opportunity_cost = min(opportunity_cost, reward * 0.8)
            reward = reward - opportunity_cost
            
            # Apply sequential bonus
            if consecutive_successes > 0:
                sequential_bonus = reward * (consecutive_success_bonus * consecutive_successes)
                reward += sequential_bonus
            
            return reward
        
        elif action_type == 'failure':
            # Standard penalty with exponential decay
            penalty = -failure_base_penalty * np.exp(-decay_rate * delay)
            
            # Add opportunity cost to penalty
            opportunity_cost = opportunity_cost_factor * delay * failure_base_penalty
            penalty = penalty - opportunity_cost
            
            return penalty
        
        elif action_type == 'missed_opportunity':
            # Enhanced penalty for missed opportunities
            missed_penalty = -failure_base_penalty * missed_opp_multiplier * np.exp(-decay_rate * delay)
            return missed_penalty
        
        elif action_type == 'no_action':
            # Reward for correctly staying out of the market
            return 100
        
        return 0
    
    def train_episode(self, episode: int) -> Dict[str, Any]:
        """
        Train a single episode using PPO.
        
        Args:
            episode: Current episode number
            
        Returns:
            Dictionary containing episode statistics
        """
        total_reward = 0
        number_trans = 0
        wins = 0
        lose = 0
        defeat = 0
        consecutive_success = 0
        
        step = 0
        next_step = 0
        
        pbar = tqdm(total=len(self.df_normalized), desc=f"Episode {episode + 1}")
        
        while step < len(self.df_normalized):
            state = self.get_state(self.df_normalized, step)
            action, log_prob, value = self.agent.select_action(state)
            done = False
            
            if action == 1:  # BUY
                next_state_row = self.target_buy.iloc[step]
                next_state_index = next_state_row["next_state_index"]
                
                reward = 0
                if next_state_row['action'] == "Target Hit":
                    wins += 1
                    consecutive_success += 1
                    reward = self.calculate_optimized_scalping_reward(
                        delay=self.target_buy.iloc[step]['delay'],
                        action_type="success",
                        consecutive_successes=consecutive_success
                    )
                elif next_state_row['action'] == "Stop Loss Hit":
                    defeat += 1
                    consecutive_success = 0
                    reward = self.calculate_optimized_scalping_reward(
                        delay=self.target_buy.iloc[step]['delay'],
                        action_type="failure",
                        consecutive_successes=consecutive_success
                    )
                elif next_state_row['action'] == "End of Day":
                    lose += 1
                    consecutive_success = 0
                    done = True
                    reward = self.calculate_optimized_scalping_reward(
                        delay=self.target_buy.iloc[step]['delay'],
                        action_type="failure",
                        consecutive_successes=consecutive_success
                    )
                
                # Store transition
                self.agent.store_transition(state, action, reward, log_prob, value, done)
                number_trans += 1
                next_step = next_state_index + 1
                
            elif action == 2:  # SELL
                next_state_row = self.target_sell.iloc[step]
                next_state_index = next_state_row["next_state_index"]
                
                reward = 0
                if next_state_row['action'] == "Target Hit":
                    wins += 1
                    consecutive_success += 1
                    reward = self.calculate_optimized_scalping_reward(
                        delay=self.target_sell.iloc[step]['delay'],
                        action_type="success",
                        consecutive_successes=consecutive_success
                    )
                elif next_state_row['action'] == "Stop Loss Hit":
                    consecutive_success = 0
                    defeat += 1
                    reward = self.calculate_optimized_scalping_reward(
                        delay=self.target_sell.iloc[step]['delay'],
                        action_type="failure",
                        consecutive_successes=consecutive_success
                    )
                elif next_state_row['action'] == "End of Day":
                    consecutive_success = 0
                    lose += 1
                    done = True
                    reward = self.calculate_optimized_scalping_reward(
                        delay=self.target_sell.iloc[step]['delay'],
                        action_type="failure",
                        consecutive_successes=consecutive_success
                    )
                
                # Store transition
                self.agent.store_transition(state, action, reward, log_prob, value, done)
                number_trans += 1
                next_step = next_state_index + 1
                
            elif action == 0:  # HOLD
                buy_side = self.target_buy.iloc[step].copy()
                sell_side = self.target_sell.iloc[step].copy()
                
                if buy_side['action'] == "Target Hit":
                    reward = self.calculate_optimized_scalping_reward(
                        delay=self.target_buy.iloc[step]['delay'],
                        action_type="missed_opportunity",
                        consecutive_successes=consecutive_success
                    )
                elif sell_side['action'] == "Target Hit":
                    reward = self.calculate_optimized_scalping_reward(
                        delay=self.target_sell.iloc[step]['delay'],
                        action_type="missed_opportunity",
                        consecutive_successes=consecutive_success
                    )
                else:
                    reward = 100
                
                # Store transition
                self.agent.store_transition(state, action, reward, log_prob, value, done)
                next_step = step + 1
            
            total_reward += reward
            
            # Update PPO policy at specified frequency
            self.step_count += 1
            if self.step_count % self.update_frequency == 0:
                self.agent.update_policy()
            
            pbar.update(next_step - step)
            step = next_step
        
        pbar.close()
        
        # Update policy at end of episode if we haven't updated recently
        if len(self.agent.memory.states) > 0:
            self.agent.update_policy()
        
        return {
            'episode': episode,
            'total_reward': total_reward,
            'number_trans': number_trans,
            'wins': wins,
            'lose': lose,
            'defeat': defeat,
            'win_rate': wins / max(number_trans, 1) * 100
        }
    
    def train(self, start_episode: int = 0, save_frequency: int = 5):
        """
        Main training loop.
        
        Args:
            start_episode: Episode to start training from (for resuming)
            save_frequency: Frequency of model saving (every N episodes)
        """
        print(f"Starting PPO training for {self.num_epochs} episodes...")
        print(f"Models will be saved every {save_frequency} episodes to {self.save_folder}")
        
        for episode in range(start_episode, self.num_epochs):
            # Train episode
            episode_stats = self.train_episode(episode)
            episode_df = pd.DataFrame([episode_stats])
            
            # Save episode statistics
            if not os.path.exists(self.stats_csv):
                episode_df.to_csv(self.stats_csv, index=False)
            else:
                episode_df.to_csv(self.stats_csv, mode='a', index=False, header=False)
            
            # Save models periodically
            if episode % save_frequency == 0:
                # Save models
                actor_save_path = os.path.join(
                    self.weights_folder, 
                    f'{self.symbol.lower()}_{self.start_date.replace("-", "_")}_actor_{episode + 1}.pth'
                )
                critic_save_path = os.path.join(
                    self.weights_folder, 
                    f'{self.symbol.lower()}_{self.start_date.replace("-", "_")}_critic_{episode + 1}.pth'
                )
                
                self.agent.save_models(actor_save_path, critic_save_path)
                
                # Print statistics
                print(f"\nEpisode {episode + 1}/{self.num_epochs}")
                print(f"Transactions: {episode_stats['number_trans']}, "
                      f"Wins: {episode_stats['wins']}, "
                      f"Losses: {episode_stats['lose']}, "
                      f"Defeats: {episode_stats['defeat']}")
                print(f"Win Rate: {episode_stats['win_rate']:.2f}%")
                print(f"Total Reward: {episode_stats['total_reward']:.2f}")
                print(f"Models saved to: {actor_save_path} and {critic_save_path}")
                
    def run_full_training_pipeline(self, additional_csv_path: str = None, start_episode: int = 0):
        """
        Run the complete training pipeline from data loading to model training.
        
        Args:
            additional_csv_path: Path to additional CSV data to include
            start_episode: Episode to start training from
        """
        print(f"Starting full PPO training pipeline for {self.symbol}")
        print(f"Training parameters:")
        print(f"  Target profit: {self.target_profit}")
        print(f"  Stop loss: {self.stop_loss}")
        print(f"  Number of epochs: {self.num_epochs}")
        print(f"  Device: {self.device}")
        print(f"  Update frequency: {self.update_frequency}")
        
        # Load and prepare data
        self.load_data(additional_csv_path)
        self.prepare_features_and_targets()
        self.normalize_data()
        
        # Initialize models
        self.initialize_models()
        
        # Start training
        self.train(start_episode)
        
        print("PPO Training completed successfully!")


# Example usage:
if __name__ == "__main__":
    # Create trainer instance
    parser = argparse.ArgumentParser(description="Train PPO Trading Agent")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--num_epochs", type=int, default=3000, help="Number of training episodes")
    parser.add_argument("--target_profit", type=float, default=1.005, help="Target profit multiplier")
    parser.add_argument("--stop_loss", type=float, default=0.99, help="Stop loss multiplier")
    parser.add_argument("--save_folder", type=str, default=None, help="Folder to save trained models")
    parser.add_argument("--actor_model_path", type=str, default=None, help="Path to pre-trained actor model")
    parser.add_argument("--critic_model_path", type=str, default=None, help="Path to pre-trained critic model")
    parser.add_argument("--norm_params_path", type=str, default=None, help="Path to normalization params")
    parser.add_argument("--additional_csv", type=str, default=None, help="Path to additional CSV data")
    parser.add_argument("--start_episode", type=int, default=0, help="Episode to resume from")
    parser.add_argument("--update_frequency", type=int, default=2048, help="PPO update frequency")
    args = parser.parse_args()

    trainer = PPOTradingTrainer(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        target_profit=args.target_profit,
        stop_loss=args.stop_loss,
        num_epochs=args.num_epochs,
        save_folder=args.save_folder,
        actor_model_path=args.actor_model_path,
        critic_model_path=args.critic_model_path,
        norm_params_path=args.norm_params_path,
        update_frequency=args.update_frequency
    )

    trainer.run_full_training_pipeline(
        additional_csv_path=args.additional_csv,
        start_episode=args.start_episode
    )

# Example command to run:
# python ppo_trainer.py --symbol SUZLON --start_date 2023-01-01 --end_date 2023-12-31 --num_epochs 3000 --update_frequency 2048

# Example command with pre-trained models:
# python ppo_trainer.py \
#     --symbol SUZLON \
#     --start_date 2025-06-15 \
#     --end_date 2025-06-19 \
#     --num_epochs 3000 \
#     --target_profit 1.005 \
#     --stop_loss 0.99 \
#     --save_folder ./trained_models/suzlon_ppo_training \
#     --actor_model_path ./trained_models/suzlon_actor.pth \
#     --critic_model_path ./trained_models/suzlon_critic.pth \
#     --norm_params_path ./json_files/suzlon_norm_params.json \
#     --additional_csv ./suzlon_2025-06-19.csv \
#     --start_episode 0 \
#     --update_frequency 1024