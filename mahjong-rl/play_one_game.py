# play_one_game.py
# 运行一局完整的麻将，输出详细对战过程

import torch
import numpy as np
from env import MahjongGBEnv
from feature import FeatureAgent
from model import CNNModel

def print_game_state(env, step_num, action_name=""):
    """打印当前牌局状态"""
    print(f"\n{'='*60}")
    print(f"第 {step_num} 步")
    if action_name:
        print(f"动作: {action_name}")
    
    for i in range(4):
        player_name = f"Player {i}"
        if i == env.curPlayer:
            player_name += " (当前)"
        
        hand = sorted(env.hands[i]) if hasattr(env, 'hands') else []
        packs = env.packs[i] if hasattr(env, 'packs') else []
        
        print(f"  {player_name}:")
        print(f"    手牌: {hand}")
        if packs:
            print(f"    副露: {packs}")
    
    if hasattr(env, 'curTile') and env.curTile:
        print(f"  当前牌: {env.curTile}")
    if hasattr(env, 'shownTiles') and env.shownTiles:
        shown = {k: v for k, v in env.shownTiles.items() if v > 0}
        if shown:
            print(f"  已出现: {shown}")


def play_game(model_path="./model-test/model_100.pt", verbose=True):
    """运行一局完整的麻将并输出详细过程"""
    
    # 加载模型
    model = CNNModel()
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        print(f"已加载模型: {model_path}")
    else:
        print("使用随机初始化模型")
    
    model.eval()
    
    # 创建环境（关闭复盘模式，更真实）
    env = MahjongGBEnv(config={
        'agent_clz': FeatureAgent,
        'duplicate': False  # 关闭复盘模式，看真实对战
    })
    
    # 初始化
    obs = env.reset()
    step_num = 0
    
    print("="*60)
    print("开始新一局麻将")
    print(f"圈风: {env.prevalentWind}")
    print(f"牌墙剩余: {len(env.tileWall) if not env.duplicate else 'N/A (复盘模式)'} 张")
    
    if verbose:
        print_game_state(env, step_num, "初始状态")
    
    done = False
    game_log = []
    
    while not done:
        # 每个玩家决策
        actions = {}
        for agent_name in obs:
            state = obs[agent_name]
            state['observation'] = torch.tensor(state['observation'], dtype=torch.float).unsqueeze(0)
            state['action_mask'] = torch.tensor(state['action_mask'], dtype=torch.float).unsqueeze(0)
            
            with torch.no_grad():
                logits, value = model(state)
                # 选择概率最高的合法动作
                mask = state['action_mask'].bool()
                masked_logits = torch.where(mask, logits, torch.tensor(-1e8))
                action = masked_logits.argmax(dim=-1).item()
            
            actions[agent_name] = action
            
            # 记录决策
            agent_idx = int(agent_name.split('_')[1]) - 1
            if verbose and agent_idx == env.curPlayer:
                action_str = env.agents[agent_idx].action2response(action)
                game_log.append(f"Step {step_num}: {agent_name} -> {action_str}")
        
        # 执行动作
        step_num += 1
        next_obs, rewards, done = env.step(actions)
        
        if verbose:
            # 获取当前玩家的动作描述
            cur_agent = f"player_{env.curPlayer + 1}"
            action_str = "未知"
            if cur_agent in actions:
                agent_idx = env.curPlayer
                action_str = env.agents[agent_idx].action2response(actions[cur_agent])
            
            print_game_state(env, step_num, f"{cur_agent} {action_str}")
            
            # 打印奖励
            if done and rewards:
                print(f"\n{'='*60}")
                print("对局结束！")
                print(f"最终得分: {rewards}")
                
                # 找出赢家
                if rewards:
                    winner = max(rewards, key=rewards.get)
                    print(f"赢家: {winner} (得分: {rewards[winner]})")
        
        obs = next_obs
    
    return rewards, game_log


if __name__ == '__main__':
    import sys
    
    # 可以通过命令行参数指定模型路径
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./model-test/model_100.pt"
    
    try:
        rewards, log = play_game(model_path, verbose=True)
        print(f"\n{'='*60}")
        print("对战结束")
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()
