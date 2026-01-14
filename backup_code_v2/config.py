"""
项目配置文件
定义常量和配置参数
"""
import os
from datetime import datetime

# 数据路径配置 - 修改为项目内的data目录
DATA_PATH = "data/HSI.xlsx"  # 使用相对路径
PREDICT_DIR = "predict_price"
GRAPH_DIR = "predict_graph"
SIGNAL_DIR = "predict_signal"
BACKTEST_DIR = "backtest_result"

# 模型配置
MODEL_NAME = "NeoQuasar/Kronos-base"
DEVICE = "cuda" if os.environ.get("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"

# 预测配置
PREDICTION_DAYS = 5  # 预测未来5天

# 交易策略配置
BUY_THRESHOLD = 0.01  # 涨幅超过1%时买入
SELL_THRESHOLD = -0.01  # 跌幅低于-1%时卖出
INITIAL_CAPITAL = 100000  # 初始资金

# 文件路径配置
OUTPUT_DIRS = [PREDICT_DIR, GRAPH_DIR, SIGNAL_DIR, BACKTEST_DIR]

# 日志配置
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 模型缓存目录
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "models")

# 创建输出目录
for directory in OUTPUT_DIRS:
    os.makedirs(directory, exist_ok=True)