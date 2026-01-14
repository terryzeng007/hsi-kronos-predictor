"""
Kronos模型加载器
用于正确加载和使用Kronos金融预测模型
"""
import torch
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class KronosModelLoader:
    """
    专门用于加载和管理Kronos模型的类
    """
    
    def __init__(self, model_name="NeoQuasar/Kronos-base", tokenizer_name="NeoQuasar/Kronos-Tokenizer-base"):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.model = None
        self.tokenizer = None
        self.predictor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        
    def load_kronos_model(self):
        """
        加载Kronos模型，如果失败则抛出异常
        """
        try:
            logger.info(f"正在加载Kronos模型: {self.model_name}")
            
            # 从Kronos源码导入必要的类
            import sys
            sys.path.insert(0, './kronos_src')
            from model import Kronos, KronosTokenizer, KronosPredictor
            
            # 加载Kronos分词器和模型
            logger.info(f"正在加载分词器: {self.tokenizer_name}")
            self.tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
            
            logger.info(f"正在加载模型: {self.model_name}")
            self.model = Kronos.from_pretrained(self.model_name)
            
            # 初始化预测器
            self.predictor = KronosPredictor(self.model, self.tokenizer, device=self.device, max_context=512)
            
            self.is_loaded = True
            logger.info("Kronos模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"加载Kronos模型失败: {str(e)}")
            raise RuntimeError(f"无法加载Kronos模型 {self.model_name}，错误信息: {str(e)}")
    
    def predict_prices(self, historical_data, prediction_days=5):
        """
        使用Kronos模型预测价格
        
        Args:
            historical_data (pd.DataFrame): 历史价格数据
            prediction_days (int): 预测天数
            
        Returns:
            pd.DataFrame: 预测结果
        """
        if not self.is_loaded:
            raise RuntimeError("Kronos模型未加载，请先调用load_kronos_model()")
        
        try:
            # 准备输入数据
            prepared_data = self._prepare_input_data(historical_data, prediction_days)
            
            # 使用Kronos预测器进行预测
            x_df, x_timestamp, y_timestamp, pred_len = prepared_data
            
            pred_df = self.predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=False
            )
            
            # 处理预测结果
            result = self._process_predictions(pred_df, y_timestamp)
            
            return result
            
        except Exception as e:
            logger.error(f"预测过程中出错: {str(e)}")
            raise RuntimeError(f"Kronos预测失败: {str(e)}")
    
    def _prepare_input_data(self, df, prediction_days):
        """
        准备输入数据用于Kronos模型
        
        Args:
            df (pd.DataFrame): 历史数据
            prediction_days (int): 预测天数
            
        Returns:
            tuple: (x_df, x_timestamp, y_timestamp, pred_len)
        """
        # 确保数据包含必需的列
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df_copy = df.copy()
        
        # 重命名列以匹配Kronos期望的格式
        column_mapping = {
            'Date': 'timestamps',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        # 检查并重命名列
        for old_col, new_col in column_mapping.items():
            if old_col in df_copy.columns:
                df_copy.rename(columns={old_col: new_col}, inplace=True)
            elif new_col not in df_copy.columns:
                # 如果缺少必需列，使用默认值
                if new_col == 'volume':
                    df_copy[new_col] = 1000000  # 默认成交量
                elif new_col in ['open', 'high', 'low', 'close']:
                    # 使用Close列的值填充其他价格列
                    if 'close' in df_copy.columns:
                        df_copy[new_col] = df_copy['close']
                    elif 'Close' in df_copy.columns:
                        df_copy[new_col] = df_copy['Close']
                    else:
                        # 如果没有收盘价列，使用最后一列
                        df_copy[new_col] = df_copy.iloc[:, -1]
        
        # 确保timestamps列存在且为datetime格式
        if 'timestamps' not in df_copy.columns:
            if 'Date' in df_copy.columns:
                df_copy['timestamps'] = pd.to_datetime(df_copy['Date'])
            else:
                # 创建一个日期范围
                df_copy['timestamps'] = pd.date_range(end=datetime.now(), periods=len(df_copy), freq='D')
        
        # 确保数值列是浮点类型
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        # 确保数据按时间排序
        df_copy = df_copy.sort_values('timestamps').reset_index(drop=True)
        
        # 限制数据长度以符合max_context限制
        max_context = 512
        if len(df_copy) > max_context:
            df_copy = df_copy.tail(max_context).reset_index(drop=True)
        
        # 定义回溯窗口和预测长度
        lookback = min(len(df_copy)-prediction_days, 400)  # 限制回溯窗口
        pred_len = prediction_days
        
        # 准备输入数据
        x_df = df_copy.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume']]
        # 如果没有amount列，添加一个默认值
        if 'amount' not in x_df.columns:
            x_df['amount'] = x_df['close'] * x_df['volume'] / 1000  # 简单估算
        
        x_timestamp = df_copy.loc[:lookback-1, 'timestamps']
        y_timestamp = df_copy.loc[lookback:lookback+pred_len-1, 'timestamps'] if len(df_copy) > lookback else pd.date_range(start=df_copy['timestamps'].iloc[-1] + timedelta(days=1), periods=pred_len, freq='D')
        
        return x_df, x_timestamp, y_timestamp, pred_len
    
    def _process_predictions(self, pred_df, y_timestamp):
        """
        处理预测结果
        
        Args:
            pred_df (pd.DataFrame): 模型预测结果
            y_timestamp (pd.Series): 预测时间戳
            
        Returns:
            pd.DataFrame: 格式化后的预测结果
        """
        # 确保预测结果与时间戳对齐
        if len(pred_df) != len(y_timestamp):
            # 如果长度不匹配，截取较短的长度
            min_len = min(len(pred_df), len(y_timestamp))
            pred_df = pred_df.head(min_len)
            y_timestamp = y_timestamp.head(min_len)
        
        # 创建结果DataFrame
        result = pd.DataFrame({
            'Date': y_timestamp.dt.strftime('%Y-%m-%d'),
            'Predicted_Open': pred_df['open'].values,
            'Predicted_High': pred_df['high'].values,
            'Predicted_Low': pred_df['low'].values,
            'Predicted_Close': pred_df['close'].values,
            'Predicted_Volume': pred_df['volume'].values.astype(int)
        })
        
        return result