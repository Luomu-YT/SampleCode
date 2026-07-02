# Mahjong-RL — 基于 PPO 的麻将 AI 强化学习训练框架

本项目使用 **PPO (Proximal Policy Optimization)** 算法训练麻将 AI，采用 **Actor-Learner 多进程架构**，支持从监督学习模型初始化权重，可提交至 Botzone 在线对战平台。

---

## 项目结构

| 文件 | 说明 |
|------|------|
| `train.py` | **训练入口**，配置参数并启动 Actor + Learner |
| `actor.py` | Actor 进程，自博弈收集对局数据 |
| `learner.py` | Learner 进程，PPO 训练更新模型 |
| `env.py` | 麻将环境，封装国标麻将规则引擎 |
| `model.py` | `CNNModel` 神经网络（策略 + 价值双分支） |
| `feature.py` | `FeatureAgent`，状态编码（6×4×9 观测 + 235 维动作掩码） |
| `agent.py` | `MahjongGBAgent` 抽象基类 |
| `model_pool.py` | 模型池，通过共享内存同步多进程模型 |
| `replay_buffer.py` | 经验回放缓冲区（Actor 生产 → Learner 消费） |
| `api_server.py` | FastAPI 服务，提供实时出牌建议 API |
| `__main__.py` | Botzone 在线评测平台提交入口 |
| `checkpoint/` | 预训练模型存放目录（如 SL 的 `15.pkl`） |
| `model-test/` | RL 训练保存的 checkpoint 目录 |

---

## 快速开始

### 1. 环境准备

依赖 PyTorch、NumPy 和 PyMahjongGB 规则引擎：

```bash
# 进入项目目录
cd mahjong-rl

# 安装依赖
pip install torch numpy fastapi uvicorn pydantic
# PyMahjongGB 需单独编译安装，详见：https://github.com/ailab-pku/PyMahjongGB
```

### 2. 启动训练

```bash
# 从预训练模型开始 RL 微调
python train.py
```

训练参数在 `train.py` 的 `config` 字典中配置，当前配置兼顾办公与训练：
- `num_actors=6`：6 个 Actor 并行自博弈
- `batch_size=128`：每轮训练采样 128 条经验
- `pretrained_model_path='./checkpoint/15.pkl'`：从 SL 模型初始化

### 3. 查看日志

```bash
# 实时显示并保存日志
python train.py | Tee-Object -FilePath training.log

# 另一个终端实时查看最新进度
Get-Content training.log -Tail 30 -Wait

# 过滤关键信息
Get-Content training.log -Wait | Select-String "Avg Reward"
Get-Content training.log -Wait | Select-String "Loss\(Policy"
```

---

## 常用指令速查

| 场景 | 指令 |
|------|------|
| 启动训练 | `python train.py` |
| 训练 + 保存日志 | `python train.py \| Tee-Object -FilePath training.log` |
| 实时查看日志 | `Get-Content training.log -Tail 30 -Wait` |
| 查看 Actor 统计 | `Select-String "Avg Reward" training.log` |
| 查看 Learner 损失 | `Select-String "Loss\(Policy" training.log` |
| 中断训练并保存 | `Ctrl + C`（会自动保存 `model_final.pt`） |

---

## 训练监控指标

### Actor 输出（每 10 局）

```
[Actor-0 ] Episode   10/1000 | Model  0 | Avg Reward:   -2.40 | Best:   24.00 | Worst:  -30.00
```

- **Avg Reward**：越高越好，说明模型得分能力在变强
- **Best/Worst**：观察波动范围，判断训练稳定性

### Learner 输出（每次迭代）

```
[Learner] Iter   150/100000 | Buffer:  12800/  28400 | Batch Reward:    -0.2341
[Learner]          | Loss(Policy/Value/Entropy/Total):   0.023456 /   0.156789 /   1.234567 /   1.414812
```

- **Buffer**：回放缓冲区已采样 / 已存入样本数
- **Policy Loss**：策略梯度损失，绝对值越小越稳定
- **Value Loss**：价值函数 MSE，越小预测越准
- **Entropy**：策略熵，太高=太随机，太低=太保守

---

## 模型文件说明

| 文件 | 来源 | 格式 |
|------|------|------|
| `checkpoint/15.pkl` | 监督学习 (SL) | `.pkl`（PyTorch state_dict） |
| `model-test/model_100.pt` | 强化学习 (RL) | `.pt`（PyTorch state_dict） |

> `.pkl` 和 `.pt` 本质相同，均为 `torch.save()` 序列化的结果，可互换使用。

---

## 从 SL 模型继续 RL 训练

`train.py` 已配置 `pretrained_model_path='./checkpoint/15.pkl'`，启动后会自动加载 SL 权重作为 RL 的初始模型。

如需修改预训练模型路径：

```python
config = {
    # ...
    'pretrained_model_path': './checkpoint/15.pkl'  # 修改这里
}
```

---

## Botzone 提交

将训练好的模型和代码打包，提交至 Botzone 在线对战平台：

```bash
# 确保 __main__.py 为入口文件，模型路径使用 data/testrl.pt
# 参考 pack_for_botzone.py 或手动打包 zip
```

---

## 项目依赖

- Python >= 3.8
- PyTorch >= 1.10
- NumPy
- PyMahjongGB（C++ 规则引擎，需单独编译）

---

## 作者

Yunlong Lu, advised by Wenxin Li, PKU AI Lab
