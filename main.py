"""
K线预测Agent主程序
功能：
1. 获取恒指历史价格数据
2. 使用Kronos模型进行K线预测
3. 回测交易策略
4. 生成可视化图表和统计结果
"""
import os
import pandas as pd
from datetime import datetime
import logging

from data_loader import DataLoader
from predictor import Predictor
from backtester import Backtester
from visualizer import Visualizer
from config import DATA_PATH  # 从配置文件导入路径


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hsi_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    logger.info("开始执行恒指K线预测任务")
    
    # 定义路径 - 使用配置文件中的路径
    data_path = DATA_PATH
    predict_dir = "predict_price"
    graph_dir = "predict_graph"  # 新增图表目录
    signal_dir = "predict_signal"
    backtest_dir = "backtest_result"
    
    # 创建输出目录
    for directory in [predict_dir, graph_dir, signal_dir, backtest_dir]:
        os.makedirs(directory, exist_ok=True)
    
    try:
        # 1. 加载数据
        logger.info("正在加载恒指历史数据...")
        data_loader = DataLoader(data_path)
        df = data_loader.load_data()
        
        # 2. 进行预测
        logger.info("正在进行K线预测...")
        predictor = Predictor()
        predictions = predictor.predict_next_days(df, days=5)
        
        # 3. 执行回测
        logger.info("正在执行回测交易...")
        backtester = Backtester(predictions)
        signals = backtester.generate_signals()
        backtest_results = backtester.run_backtest()
        
        # 4. 生成可视化图表
        logger.info("正在生成可视化图表...")
        visualizer = Visualizer(df, predictions, signals, backtest_results)
        visualizer.create_prediction_chart(graph_dir)  # 修复：使用正确的图表目录
        visualizer.create_signals_chart(signal_dir)
        visualizer.create_backtest_summary(backtest_dir)
        
        logger.info("恒指K线预测任务完成！")
        
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()