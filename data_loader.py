"""
数据加载模块
负责从本地文件加载恒指历史价格数据
"""
import pandas as pd
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, file_path):
        """
        初始化数据加载器
        
        Args:
            file_path (str): Excel文件路径
        """
        self.file_path = file_path
        
    def load_data(self):
        """
        加载Excel中的恒指历史数据
        
        Returns:
            pd.DataFrame: 包含历史价格数据的数据框
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"找不到数据文件: {self.file_path}")
        
        try:
            # 读取Excel文件
            df = pd.read_excel(self.file_path)
            
            # 假设数据包含日期、开盘价、最高价、最低价、收盘价、成交量等列
            # 如果列名不同，需要相应调整
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            # 检查必要列是否存在
            for col in required_columns:
                if col not in df.columns:
                    logger.warning(f"警告: 列 '{col}' 不存在于数据中")
            
            # 确保日期列为datetime类型
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')  # 按日期排序
            
            logger.info(f"成功加载 {len(df)} 条历史数据")
            return df
            
        except Exception as e:
            logger.error(f"加载数据时出错: {str(e)}")
            raise
    
    def get_latest_data(self, n_days=1):
        """
        获取最近n天的数据
        
        Args:
            n_days (int): 获取最近几天的数据
            
        Returns:
            pd.DataFrame: 最近n天的数据
        """
        df = self.load_data()
        return df.tail(n_days).reset_index(drop=True)