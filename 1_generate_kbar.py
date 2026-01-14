"""
使用Kronos模型生成K线数据并绘制K线图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import sys
import os

# 添加Kronos源码路径
sys.path.insert(0, './kronos_src')

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    KRONOS_AVAILABLE = True
except ImportError:
    print("警告: 未找到Kronos模型")
    KRONOS_AVAILABLE = False

import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_real_data(file_path="./data/HSI_2.xlsx"):
    """
    加载真实的历史数据
    
    Args:
        file_path: Excel文件路径
    
    Returns:
        DataFrame: 包含OHLCV数据的DataFrame
    """
    try:
        df = pd.read_excel(file_path)
        
        # 检查必要的列是否存在
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                print(f"警告: 列 '{col}' 不存在于数据中")
        
        # 重命名列以匹配Kronos期望的格式
        column_mapping = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        # 重命名现有列
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # 确保timestamps列存在且为datetime格式
        if 'date' not in df.columns:
            if 'Date' in df.columns:
                df['date'] = pd.to_datetime(df['Date'])
            else:
                # 如果没有日期列，创建一个日期范围
                df['date'] = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
        else:
            df['date'] = pd.to_datetime(df['date'])
        
        # 确保数值列是浮点类型
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 移除包含NaN值的行
        df.dropna(inplace=True)
        
        # 确保数据按时间排序
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"成功加载 {len(df)} 条真实历史数据")
        return df
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return None
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None


def generate_kline_with_kronos(real_data, num_points=100):
    """
    使用Kronos模型基于真实数据生成K线数据
    
    Args:
        real_data: 真实的历史数据
        num_points: 要生成的数据点数量
    
    Returns:
        DataFrame: 生成的K线数据
    """
    if not KRONOS_AVAILABLE:
        print("错误: Kronos模型不可用")
        return None
    
    if real_data is None or len(real_data) == 0:
        print("错误: 没有有效的输入数据")
        return None
    
    try:
        print("正在加载Kronos模型...")
        
        # 加载预训练的Kronos模型和分词器
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")  # 使用small版本以节省资源
        
        # 初始化预测器
        predictor = KronosPredictor(model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # 准备预测所需的数据
        lookback = min(400, len(real_data))  # 使用最多400个历史数据点
        pred_len = num_points  # 预测未来的num_points个数据点
        
        # 选择最近的lookback个数据点
        x_df = real_data.tail(lookback)[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # 如果缺少amount列，添加一个默认值
        if 'amount' not in x_df.columns:
            x_df['amount'] = x_df['close'] * x_df['volume'] / 1000  # 简单估算
        
        x_timestamp = real_data.tail(lookback)['date']
        
        # 创建未来的时间戳
        last_date = real_data['date'].iloc[-1]
        y_timestamp = pd.date_range(start=last_date + timedelta(days=1), periods=pred_len, freq='D')
        y_timestamp = pd.DataFrame(y_timestamp, columns=['date'])
        y_timestamp['date'] = pd.to_datetime(y_timestamp['date'])
        y_timestamp = y_timestamp['date']
        print(f"使用Kronos模型基于 {lookback} 天历史数据预测未来 {pred_len} 天的数据...")
        
        # 使用Kronos进行预测
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,  # 温度参数
            top_p=0.9,  # nucleus采样概率
            sample_count=1  # 采样次数
        )
        
        # 将预测结果与时间戳合并
        result_df = pd.DataFrame({
            'date': y_timestamp,
            'open': pred_df['open'].values,
            'high': pred_df['high'].values,
            'low': pred_df['low'].values,
            'close': pred_df['close'].values,
            'volume': pred_df['volume'].values
        })
        
        print("Kronos预测完成!")
        return result_df
        
    except Exception as e:
        print(f"使用Kronos模型时出错: {e}")
        return None



def plot_matplotlib_kline(df, title="K线图"):
    """
    使用matplotlib绘制K线图，生成两种样式：
    1. 带坐标轴的完整图
    2. 纯净的只含K线的图
    
    Args:
        df: 包含OHLC数据的DataFrame
        title: 图表标题
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        from matplotlib.patches import Rectangle
        import matplotlib.dates as mdates
        
        # 设置中文字体
        rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False
        
        # 第一种图：带坐标轴的完整图
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        
        # 绘制K线
        for idx, (_, row) in enumerate(df.iterrows()):
            date = mdates.date2num(row['date'])
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # 判断涨跌
            color = 'red' if close_price >= open_price else 'green'
            
            # 绘制最高价到最低价的线
            ax1.plot([date, date], [low_price, high_price], color=color, linewidth=0.5)
            
            # 绘制开盘价到收盘价的实体
            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)
            rect = Rectangle((date-0.3, bottom), 0.6, height, facecolor=color, edgecolor=color)
            ax1.add_patch(rect)
        
        ax1.set_title(title, fontsize=14)
        ax1.set_ylabel('价格', fontsize=12)
        ax1.set_facecolor('white')  # 白色背景
        ax1.grid(False)  # 无网格线
        
        # 格式化x轴
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"predict_graph/kline_with_axes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("带坐标轴的K线图已保存")
        
        # 第二种图：纯净的只含K线的图
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        
        # 绘制K线
        for idx, (_, row) in enumerate(df.iterrows()):
            date = mdates.date2num(row['date'])
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # 判断涨跌
            color = 'red' if close_price >= open_price else 'green'
            
            # 绘制最高价到最低价的线
            ax2.plot([date, date], [low_price, high_price], color=color, linewidth=0.5)
            
            # 绘制开盘价到收盘价的实体
            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)
            rect = Rectangle((date-0.3, bottom), 0.6, height, facecolor=color, edgecolor=color)
            ax2.add_patch(rect)
        
        # 设置纯净样式
        ax2.set_facecolor('white')  # 白色背景
        ax2.grid(False)  # 无网格线
        ax2.set_xticks([])  # 移除x轴刻度
        ax2.set_yticks([])  # 移除y轴刻度
        ax2.spines['top'].set_visible(False)  # 隐藏顶部边框
        ax2.spines['right'].set_visible(False)  # 隐藏右侧边框
        ax2.spines['bottom'].set_visible(False)  # 隐藏底部边框
        ax2.spines['left'].set_visible(False)  # 隐藏左侧边框
        
        plt.tight_layout()
        plt.savefig(f"predict_graph/kline_pure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print("纯净K线图已保存")
        
    except ImportError:
        print("matplotlib未安装，跳过matplotlib绘图")


def main():
    """主函数"""
    print("开始加载真实数据...")
    
    # 加载真实数据
    real_data = load_real_data("./data/HSI.xlsx")
    
    if real_data is None or len(real_data) == 0:
        print("无法加载真实数据，程序退出")
        return
    
    print(f"数据预览 (前10行):")
    print(real_data.head(10))
    
    # 使用Kronos模型基于真实数据生成K线数据
    print("\n开始使用Kronos模型生成K线数据...")
    kline_data = generate_kline_with_kronos(real_data, num_points=50)  # 生成50个数据点
    
    if kline_data is None or len(kline_data) == 0:
        print("Kronos模型未能生成有效数据")
        return
    
    print(f"生成了 {len(kline_data)} 条预测K线数据")
    print("预测数据预览 (前10行):")
    print(kline_data.head(10))
    
    # 使用matplotlib绘制（生成两种样式）
    plot_matplotlib_kline(kline_data, "K线图 (matplotlib)")
    
    # 保存数据到CSV
    csv_filename = f"predict_price/generated_kline_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    kline_data.to_csv(csv_filename, index=False)
    print(f"\n生成的K线数据已保存到: {csv_filename}")


if __name__ == "__main__":
    main()