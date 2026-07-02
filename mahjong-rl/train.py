# Yunlong Lu created at 2022/11/22, advised by Wenxin Li, PKU AI Lab

from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner

if __name__ == '__main__':
    # config = {
    #     'replay_buffer_size': 50000,
    #     'replay_buffer_episode': 400,
    #     'model_pool_size': 20,
    #     'model_pool_name': 'model-pool',
    #     'num_actors': 24,
    #     'episodes_per_actor': 1000,
    #     'gamma': 0.98,
    #     'lambda': 0.95,
    #     'min_sample': 200,
    #     'batch_size': 256,
    #     'epochs': 5,
    #     'clip': 0.2,
    #     'lr': 1e-4,
    #     'value_coeff': 1,
    #     'entropy_coeff': 0.01,
    #     'device': 'cuda',
    #     'ckpt_save_interval': 300,
    #     'ckpt_save_path': '/model/'
    # }

    config = {
        'replay_buffer_size': 30000,      # 缩减，留出内存给 IDEA
        'replay_buffer_episode': 200,
        'model_pool_size': 10,           # 够用就行
        'model_pool_name': 'model-pool-test',
        'num_actors': 6,                  # 16线程留一半给系统和办公
        'episodes_per_actor': 1000,
        'gamma': 0.98,
        'lambda': 0.95,
        'min_sample': 200,
        'batch_size': 128,                # 降低，减少训练时的内存和CPU峰值
        'epochs': 5,
        'clip': 0.2,
        'lr': 1e-4,
        'value_coeff': 1,
        'entropy_coeff': 0.01,
        'device': 'cpu',                  # 使用 CPU
        'ckpt_save_interval': 300,
        'ckpt_save_path': './model/test5/', # 本地路径
        'max_iterations': 100000,          # 最大训练轮数
        'save_interval': 1000,             # 每1000轮保存一次
        'pretrained_model_path': './model/test4/model_4000.pt'  # 从监督学习模型初始化
    }


    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    
    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer)
        actors.append(actor)
    learner = Learner(config, replay_buffer)
    
    for actor in actors: actor.start()
    learner.start()
    
    for actor in actors: actor.join()
    learner.terminate()