import gym
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        
        self.df = df
        self.current_step = 0
        self.prev_step =0
        self.balance = 1000000
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.max_steps = len(df) - 1
        
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(18,), dtype=np.float32)

        self.No_of_transactions =0
        
    
    def reset(self):
        self.current_step = 0
        self.prev_step =0
        self.balance = 1000000
        
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.No_of_transactions =0

        return self._get_observation()
    
    def _get_observation(self):
        return np.array([
          
            self.df.iloc[self.current_step]['close'],
            self.df.iloc[self.current_step]['high'],
            self.df.iloc[self.current_step]['low'],
            self.df.iloc[self.current_step]['trades'],
            self.df.iloc[self.current_step]['open'],
            self.df.iloc[self.current_step]['time'],
            self.df.iloc[self.current_step]['volume'],
            self.df.iloc[self.current_step]['vwap'],
            self.df.iloc[self.current_step]['MA50'],
            self.df.iloc[self.current_step]['RSI'],
            self.df.iloc[self.current_step]['MACD'],
            self.df.iloc[self.current_step]['BB_upper'],
            self.df.iloc[self.current_step]['BB_lower'],
            self.df.iloc[self.current_step]['ADX'],
            self.df.iloc[self.current_step]['CCI'],
            self.df.iloc[self.current_step]['ATR'],
            self.df.iloc[self.current_step]['ROC'],
            self.df.iloc[self.current_step]['OBV']
        
        ])
    def _get_primary_observations(self):
        return np.array([
            self.balance,
            self.shares_held,
            self.No_of_transactions,      
        ])
    
    # def step(self, action):
    #     current_price = self.df.iloc[self.current_step]['close']
      
    #     prev_val = self.balance + self.shares_held*self.df.iloc[self.prev_step]['close']
        
    #     reward =0
    #     if action == 1:  # Buy
    #         self.shares_held += self.balance // current_price
    #         self.balance %= current_price
    #         self.prev_step = self.current_step
    #         reward =0
        
    #     elif action == 2:  # Sell
    #         if(self.shares_held!=0):
    #             self.No_of_transactions +=1
    #         self.balance += self.shares_held * current_price
    #         self.total_shares_sold += self.shares_held
    #         self.total_sales_value += self.shares_held * current_price
    #         self.shares_held = 0
    #         reward = self.balance  -prev_val
        
    #     self.current_step += 1
    #     done = self.current_step >= self.max_steps
    #     # reward = self.balance + self.shares_held * current_price + self.total_sales_value
       
        
    #     return self._get_observation(), reward, done, {}
    
    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Total sales value: {self.total_sales_value}')