import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv


class StockPortfolioEnvForMlpAgent(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then 
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step
        

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                df,
                stock_dim,
                hmax,
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold=None,
                lookback=252,
                first_day = 3,
                price_hisotry_window = 4,
                episodic_window = 4,
                obs_feature_list = ['open', 'high', 'low', 'close', 'volume', 'day']):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        assert first_day >= price_hisotry_window - 1, "first_day >= price_hisotry_window - 1"
        self.first_day = first_day
        self.day = first_day
        self.cur_epi_start_day = first_day
        self.lookback=lookback
        self.episodic_window = episodic_window
        self.price_hisotry_window = price_hisotry_window
        self.df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'tic', 'day']]
        self.obs_feature_list = obs_feature_list
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low = 0, high = 1,shape = (self.action_space,)) 

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.stock_dim*self.price_hisotry_window,len(self.obs_feature_list)))

        # load data from a pandas dataframe
        # data = df.loc[0:3,:]
        self.data = self.df.loc[self.day + 1 - self.price_hisotry_window : self.day,:]

        self.state =  np.array(self.data)
        self.terminal = False     
        self.turbulence_threshold = turbulence_threshold        
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]]

        
    def step(self, actions):
        #save last day's data
        last_day_memory = self.data

        #move to next day and get new obs
        self.day += 1
        print("day is", self.day)
        self.data = self.df.loc[self.day,:]
        self.state =  np.array(self.df[self.obs_feature_list].loc[self.day + 1 - self.price_hisotry_window : self.day,:])
        #print(self.state)

        #reshape and softmax action
        actions = actions.reshape(-1)
        #print("Model actions: ",actions)
        weights = self.softmax_normalization(actions) 
        #print("Normalized actions: ", weights)
        self.actions_memory.append(weights)
        
        # calcualte portfolio return
        # individual stocks' return * weight
        portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights)
        # update portfolio value
        new_portfolio_value = self.portfolio_value*(1+portfolio_return)
        self.portfolio_value = new_portfolio_value

        # save into memory
        self.portfolio_return_memory.append(portfolio_return)
        self.date_memory.append(self.data.date.unique()[0])            
        self.asset_memory.append(new_portfolio_value)

        # the reward is the new portfolio value or end portfolo value
        self.reward = new_portfolio_value 
        #print("Step reward: ", self.reward)
        #self.reward = self.reward*self.reward_scaling

        
        self.terminal = self.day - self.cur_epi_start_day >= self.episodic_window 

        info = []
        if(self.terminal):
            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))           
            print("end_total_asset:{}".format(self.portfolio_value))
            episodic_return = self.portfolio_value - self.asset_memory[0] 
            episodic_length = self.day - self.cur_epi_start_day
            info.append({"episode": {"r": episodic_return, "l": episodic_length}})
            self.cur_epi_start_day = self.day
        return self.state, self.reward, np.array([self.terminal]), info

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = self.first_day
        self.data = self.df.loc[self.day,:]
        self.state =  np.array(self.df[self.obs_feature_list].loc[self.day + 1 - self.price_hisotry_window : self.day,:])

        self.portfolio_value = self.initial_amount
        #self.cost = 0
        #self.trades = 0
        self.terminal = False 
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]] 
        return self.state
    
    def render(self, mode='human'):
        return self.state
        
    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator/denominator
        return softmax_output

    
    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        #print(len(date_list))
        #print(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']
        
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs