# mahjong-rl/api_server.py
# FastAPI 接口服务，用于实时麻将AI出牌建议

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
import numpy as np
import uvicorn
from collections import Counter

from sympy.codegen.ast import break_

from model import CNNModel
from feature import FeatureAgent

# 导入胡牌计算
try:
    from MahjongGB import MahjongFanCalculator
    MAHJONG_GB_AVAILABLE = True
except ImportError:
    MAHJONG_GB_AVAILABLE = False
    print("警告: MahjongGB 库未安装，胡牌检测功能将不可用")

app = FastAPI(title="麻将AI API", version="1.1.0")

# ============== 数据模型定义 ==============

class TileInfo(BaseModel):
    """单张牌信息"""
    suit: str  # W(万), T(筒), B(条), F(风), J(箭)
    number: int  # 1-9 或 1-4(风) 或 1-3(箭)

    def to_str(self) -> str:
        return f"{self.suit}{self.number}"

class PackInfo(BaseModel):
    """副露信息（吃、碰、杠）"""
    type: str  # CHI, PENG, GANG
    tile: str  # 牌，如 "W3"
    offer: int  # 0=自己，1=上家，2=对家，3=下家

class GameState(BaseModel):
    """牌局状态"""
    seat_wind: int  # 0-3，自家风位 (0=东)
    prevalent_wind: int  # 0-3，圈风
    hand: List[str]  # 手牌列表，如 ["W1", "W2", "T3"]
    packs: Optional[List[PackInfo]] = []  # 自家副露
    shown_tiles: Optional[Dict[str, int]] = {}  # 已出现的牌及数量
    tile_wall_count: Optional[int] = 70  # 剩余牌墙数量
    is_wall_last: Optional[bool] = False  # 是否牌墙最后
    history: Optional[Dict[int, List[str]]] = {}  # 各家出牌历史

class ActionRequest(BaseModel):
    """动作请求"""
    game_state: GameState
    action_type: str  # "draw"(摸牌), "play"(别人出牌), "bugang"(别人补杠)
    current_tile: Optional[str] = None  # 当前牌（摸到的或别人打的）
    is_about_kong: Optional[bool] = False  # 是否杠后摸牌
    min_fan: Optional[int] = 8  # 最小胡牌番数，默认8番（国标），设为0则无限制

class FanDetail(BaseModel):
    """番种详情"""
    name: str  # 番种名称（中文）
    name_en: str  # 番种名称（英文）
    fan: int  # 番数
    count: int  # 数量

class HuCheckResponse(BaseModel):
    """胡牌检测结果"""
    can_hu: bool  # 是否可以胡牌
    total_fan: int  # 总番数
    fans: List[FanDetail]  # 番种详情列表
    is_self_drawn: bool  # 是否自摸
    is_about_kong: bool  # 是否杠上开花/抢杠
    is_wall_last: bool  # 是否海底捞月

class ActionResponse(BaseModel):
    """动作响应"""
    action: str  # 动作类型: Pass, Hu, Play, Chi, Peng, Gang, AnGang, BuGang
    tile: Optional[str] = None  # 涉及的牌
    details: Optional[str] = None  # 详细说明
    confidence: Optional[float] = None  # 模型置信度（概率）
    hu_info: Optional[HuCheckResponse] = None  # 胡牌信息（当action为Hu时）

# ============== 全局模型加载 ==============

MODEL_PATH = "./model-test/model_final.pt"  # 可修改为其他模型路径
device = torch.device("cpu")  # 可根据需要改为 cuda

# 加载模型
model = CNNModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print(f"模型已加载: {MODEL_PATH}")

# ============== 辅助函数 ==============

def create_agent_from_state(state: GameState) -> FeatureAgent:
    """从游戏状态创建FeatureAgent"""
    agent = FeatureAgent(state.seat_wind)

    # 设置圈风
    agent.request2obs(f"Wind {state.prevalent_wind}")

    # 设置手牌
    hand_str = " ".join(state.hand)
    agent.request2obs(f"Deal {hand_str}")

    # 设置副露
    agent.packs[0] = [(p.type, p.tile, p.offer) for p in state.packs]

    # 设置其他状态
    agent.shownTiles = Counter(state.shown_tiles or {})
    agent.tileWall = [state.tile_wall_count] * 4
    agent.wallLast = state.is_wall_last
    
    # 调试信息
    print(f"Debug create_agent - Deal后hand ({len(agent.hand)}张): {agent.hand}")

    return agent

def get_action_confidence(logits: torch.Tensor, action: int) -> float:
    """获取动作的置信度（概率）"""
    probs = torch.softmax(logits, dim=-1)
    return probs[0][action].item()

def check_hu(
    hand: List[str],
    packs: List[tuple],
    win_tile: str,
    seat_wind: int,
    prevalent_wind: int,
    is_self_drawn: bool = False,
    is_about_kong: bool = False,
    is_wall_last: bool = False,
    shown_tiles: Dict[str, int] = None,
    min_fan: int = 8
) -> HuCheckResponse:
    """
    检查是否可以胡牌，返回番数详情
    """
    if not MAHJONG_GB_AVAILABLE:
        return HuCheckResponse(
            can_hu=False,
            total_fan=0,
            fans=[],
            is_self_drawn=is_self_drawn,
            is_about_kong=is_about_kong,
            is_wall_last=is_wall_last
        )

    try:
        # 确保 shown_tiles 不为 None
        if shown_tiles is None:
            shown_tiles = {}
        
        fans = MahjongFanCalculator(
            pack=tuple(packs),
            hand=tuple(hand),
            winTile=win_tile,
            flowerCount=0,
            isSelfDrawn=is_self_drawn,
            is4thTile=(shown_tiles.get(win_tile, 0) + is_self_drawn) == 4,
            isAboutKong=is_about_kong,
            isWallLast=is_wall_last,
            seatWind=seat_wind,
            prevalentWind=prevalent_wind,
            verbose=True
        )

        fan_list = []
        total_fan = 0
        for fan_point, cnt, fan_name, fan_name_en in fans:
            fan_list.append(FanDetail(
                name=fan_name,
                name_en=fan_name_en,
                fan=fan_point,
                count=cnt
            ))
            total_fan += fan_point * cnt

        # 根据配置的最小番数判断是否可胡
        can_hu = total_fan >= min_fan

        return HuCheckResponse(
            can_hu=can_hu,
            total_fan=total_fan,
            fans=fan_list,
            is_self_drawn=is_self_drawn,
            is_about_kong=is_about_kong,
            is_wall_last=is_wall_last
        )

    except Exception as e:
        # 返回错误信息以便调试
        print(f"胡牌计算异常: {e}")
        print(f"  hand: {hand}")
        print(f"  packs: {packs}")
        print(f"  win_tile: {win_tile}")
        return HuCheckResponse(
            can_hu=False,
            total_fan=0,
            fans=[FanDetail(name="计算错误", name_en=str(e), fan=0, count=1)],
            is_self_drawn=is_self_drawn,
            is_about_kong=is_about_kong,
            is_wall_last=is_wall_last
        )

# ============== API 端点 ==============

@app.post("/predict", response_model=ActionResponse)
async def predict_action(request: ActionRequest):
    """
    预测当前最佳动作，如果建议胡牌，会返回详细的番数信息
    """
    try:
        # 创建agent
        agent = create_agent_from_state(request.game_state)

        # 构建请求字符串
        if request.action_type == "draw":
            obs = agent.request2obs(f"Draw {request.current_tile}")
        elif request.action_type == "play":
            player = 1
            agent.request2obs(f"Player {player} Draw")
            obs = agent.request2obs(f"Player {player} Play {request.current_tile}")
        elif request.action_type == "bugang":
            player = 1
            obs = agent.request2obs(f"Player {player} BuGang {request.current_tile}")
        else:
            raise HTTPException(status_code=400, detail=f"未知的action_type: {request.action_type}")

        # 模型推理
        with torch.no_grad():
            input_dict = {
                'observation': torch.from_numpy(np.expand_dims(obs['observation'], 0)).float().to(device),
                'action_mask': torch.from_numpy(np.expand_dims(obs['action_mask'], 0)).float().to(device)
            }
            logits, value = model(input_dict)

            mask = input_dict['action_mask'].bool()
            masked_logits = torch.where(mask, logits, torch.tensor(-1e8).to(device))
            action = masked_logits.argmax(dim=-1).item()
            confidence = get_action_confidence(masked_logits, action)

        # 转换为响应
        response_str = agent.action2response(action)
        response_parts = response_str.split()
        action_type = response_parts[0]
        tile = response_parts[1] if len(response_parts) > 1 else None

        # 如果是胡牌，计算番数详情
        hu_info = None
        if action_type == "Hu" and request.current_tile:
            is_self_drawn = request.action_type == "draw"
            hu_info = check_hu(
                hand=agent.hand,
                packs=agent.packs[0],
                win_tile=request.current_tile,
                seat_wind=request.game_state.seat_wind,
                prevalent_wind=request.game_state.prevalent_wind,
                is_self_drawn=is_self_drawn,
                is_about_kong=request.is_about_kong,
                is_wall_last=request.game_state.is_wall_last,
                shown_tiles=agent.shownTiles,
                min_fan=request.min_fan
            )

        return ActionResponse(
            action=action_type,
            tile=tile,
            details=f"模型建议: {response_str}, 状态价值: {value.item():.4f}",
            confidence=confidence,
            hu_info=hu_info
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check_hu", response_model=HuCheckResponse)
async def check_hu_endpoint(request: ActionRequest):
    """
    独立接口：检查当前是否可以胡牌，返回详细番数信息
    
    示例请求:
    {
        "game_state": {
            "seat_wind": 0,
            "prevalent_wind": 0,
            "hand": ["W1","W2","W3","W4","W5","W6","W7","W8","W9","T1","T1","F1","F1"],
            "packs": [],
            "is_wall_last": false
        },
        "action_type": "draw",
        "current_tile": "F1",
        "is_about_kong": false
    }
    """
    try:
        # 直接使用传入的hand，不通过FeatureAgent处理
        # MahjongFanCalculator 要求 hand 不含 winTile（13张）
        hand = request.game_state.hand.copy()
        
        print(f"Debug - 原始hand ({len(hand)}张): {hand}")
        print(f"Debug - win_tile: {request.current_tile}")
        print(f"Debug - win_tile在hand中出现次数: {hand.count(request.current_tile)}")

        # 如果hand不足13张，说明输入的hand本身不含和牌，需要添加和牌到hand中
        # 如果hand超过13张，说明输入有问题
        if len(hand) != 13:
            print(f"警告: hand数量应为13张，实际为{len(hand)}张")
        
        is_self_drawn = request.action_type == "draw"
        
        # 转换packs格式
        packs = [(p.type, p.tile, p.offer) for p in request.game_state.packs]
        
        print(f"Debug - hand ({len(hand)}张): {hand}")
        print(f"Debug - win_tile: {request.current_tile}")
        print(f"Debug - packs: {packs}")
        
        return check_hu(
            hand=hand,
            packs=packs,
            win_tile=request.current_tile,
            seat_wind=request.game_state.seat_wind,
            prevalent_wind=request.game_state.prevalent_wind,
            is_self_drawn=is_self_drawn,
            is_about_kong=request.is_about_kong,
            is_wall_last=request.game_state.is_wall_last,
            shown_tiles=request.game_state.shown_tiles or {},
            min_fan=request.min_fan
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(requests: List[ActionRequest]):
    """批量预测"""
    results = []
    for req in requests:
        result = await predict_action(req)
        results.append(result)
    return results

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH,
        "device": str(device),
        "mahjong_gb_available": MAHJONG_GB_AVAILABLE
    }

@app.get("/tiles")
async def get_tile_list():
    """获取所有牌的列表"""
    return {
        "wan": [f"W{i}" for i in range(1, 10)],
        "tong": [f"T{i}" for i in range(1, 10)],
        "tiao": [f"B{i}" for i in range(1, 10)],
        "feng": [f"F{i}" for i in range(1, 5)],
        "jian": [f"J{i}" for i in range(1, 4)]
    }

# ============== 启动服务 ==============

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
