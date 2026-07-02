# Yunlong Lu created at 2022/11/22, advised by Wenxin Li, PKU AI Lab

from multiprocessing import Process
import numpy as np
import torch

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import MahjongGBEnv
from feature import FeatureAgent
from model import CNNModel

class Actor(Process):
    
    def __init__(self, config, replay_buffer):
        super(Actor, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.name = config.get('name', 'Actor-?')
        
    def run(self):
        torch.set_num_threads(1)
    
        # connect to model pool
        model_pool = ModelPoolClient(self.config['model_pool_name'])
        
        # create network model
        model = CNNModel()
        
        # load initial model
        version = model_pool.get_latest_model()
        state_dict = model_pool.load_model(version)
        model.load_state_dict(state_dict)
        
        # collect data
        env = MahjongGBEnv(config = {'agent_clz': FeatureAgent})
        policies = {player : model for player in env.agent_names} # all four players use the latest model
        
        # 用于统计最近10局的奖励
        recent_rewards = []
        
        for episode in range(self.config['episodes_per_actor']):
            # update model
            latest = model_pool.get_latest_model()
            if latest['id'] > version['id']:
                state_dict = model_pool.load_model(latest)
                model.load_state_dict(state_dict)
                version = latest
            
            # run one episode and collect data
            obs = env.reset()
            episode_data = {agent_name: {
                'state' : {
                    'observation': [],
                    'action_mask': []
                },
                'action' : [],
                'reward' : [],
                'value' : []
            } for agent_name in env.agent_names}
            done = False
            while not done:
                # each player take action
                actions = {}
                values = {}
                for agent_name in obs:
                    agent_data = episode_data[agent_name]
                    state = obs[agent_name]
                    agent_data['state']['observation'].append(state['observation'])
                    agent_data['state']['action_mask'].append(state['action_mask'])
                    state['observation'] = torch.tensor(state['observation'], dtype = torch.float).unsqueeze(0)
                    state['action_mask'] = torch.tensor(state['action_mask'], dtype = torch.float).unsqueeze(0)
                    model.train(False) # Batch Norm inference mode
                    with torch.no_grad():
                        logits, value = model(state)
                        action_dist = torch.distributions.Categorical(logits = logits)
                        action = action_dist.sample().item()
                        value = value.item()
                    actions[agent_name] = action
                    values[agent_name] = value
                    agent_data['action'].append(actions[agent_name])
                    agent_data['value'].append(values[agent_name])
                # interact with env
                next_obs, rewards, done = env.step(actions)
                for agent_name in rewards:
                    episode_data[agent_name]['reward'].append(rewards[agent_name])
                obs = next_obs
            # 记录当前 Actor 对应的主玩家 (player_1) 的得分
            # 注意：麻将是零和博弈，4家得分之和为0，不能取sum
            # env.py 中 agent_names = ['player_1', 'player_2', 'player_3', 'player_4']
            main_player_reward = rewards.get('player_1', 0)
            recent_rewards.append(main_player_reward)
            
            # 每10局打印一次统计信息
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                best_reward = max(recent_rewards) if recent_rewards else 0
                worst_reward = min(recent_rewards) if recent_rewards else 0
                print('[%-8s] Episode %4d/%4d | Model %2d | Avg Reward: %7.2f | Best: %7.2f | Worst: %7.2f' % (
                    self.name, episode + 1, self.config['episodes_per_actor'], 
                    latest['id'], avg_reward, best_reward, worst_reward))
                recent_rewards = []
            else:
                # 非统计轮次也打印简要信息（只显示主玩家得分）
                print('[%-8s] Episode %4d/%4d | Model %2d | Main Player Reward: %.2f' % (
                    self.name, episode + 1, self.config['episodes_per_actor'], 
                    latest['id'], main_player_reward))
            
            # postprocessing episode data for each agent
            for agent_name, agent_data in episode_data.items():
                if len(agent_data['action']) < len(agent_data['reward']):
                    agent_data['reward'].pop(0)
                obs = np.stack(agent_data['state']['observation'])
                mask = np.stack(agent_data['state']['action_mask'])
                actions = np.array(agent_data['action'], dtype = np.int64)
                rewards = np.array(agent_data['reward'], dtype = np.float32)
                values = np.array(agent_data['value'], dtype = np.float32)
                next_values = np.array(agent_data['value'][1:] + [0], dtype = np.float32)
                
                td_target = rewards + next_values * self.config['gamma']
                td_delta = td_target - values
                advs = []
                adv = 0
                for delta in td_delta[::-1]:
                    adv = self.config['gamma'] * self.config['lambda'] * adv + delta
                    advs.append(adv) # GAE
                advs.reverse()
                advantages = np.array(advs, dtype = np.float32)
                
                # send samples to replay_buffer (per agent)
                self.replay_buffer.push({
                    'state': {
                        'observation': obs,
                        'action_mask': mask
                    },
                    'action': actions,
                    'adv': advantages,
                    'target': td_target
                })