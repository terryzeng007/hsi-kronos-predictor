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


def load_real_data(file_path="./data/HSI.xlsx"):
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
            'Date': 'timestamps',
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
        if 'timestamps' not in df.columns:
            if 'Date' in df.columns:
                df['timestamps'] = pd.to_datetime(df['Date'])
            else:
                # 如果没有日期列，创建一个日期范围
                df['timestamps'] = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
        else:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        
        # 确保数值列是浮点类型
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 移除包含NaN值的行
        df.dropna(inplace=True)
        
        # 确保数据按时间排序
        df = df.sort_values('timestamps').reset_index(drop=True)
        
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
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")  # 使用small版本以节省资源
        
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
        
        x_timestamp = real_data.tail(lookback)['timestamps'].reset_index(drop=True)
        
        # 创建未来的时间戳
        last_date = real_data['timestamps'].iloc[-1]
        y_timestamp = pd.date_range(start=last_date + timedelta(days=1), periods=pred_len, freq='D')
        
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
            'timestamps': y_timestamp,
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


def plot_kline_chart(df, title="K线图"):
    """
    绘制K线图
    
    Args:
        df: 包含OHLCV数据的DataFrame
        title: 图表标题
    """
    # 创建交互式图表
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(title, '成交量'),
        row_width=[0.7, 0.3]
    )
    
    # 添加K线图
    fig.add_trace(
        go.Candlestick(
            x=df['timestamps'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线'
        ),
        row=1, col=1
    )
    
    # 添加成交量柱状图
    fig.add_trace(
        go.Bar(
            x=df['timestamps'],
            y=df['volume'],
            name='成交量',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    # 更新布局
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False
    )
    
    # 显示图表
    fig.show()
    
    # 同时保存为HTML文件
    filename = f"kline_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    fig.write_html(filename)
    print(f"K线图已保存为: {filename}")


def plot_matplotlib_kline(df, title="K线图"):
    """
    使用matplotlib绘制K线图（备用方法）
    
    Args:
        df: 包含OHLC数据的DataFrame
        title: 图表标题
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        from matplotlib.patches import Rectangle
        
        # 设置中文字体
        rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # 绘制K线
        for idx, (_, row) in enumerate(df.iterrows()):
            date = mdates.date2num(row['timestamps'])
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
        
        ax1.set_title(title)
        ax1.set_ylabel('价格')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 绘制成交量
        ax2.bar(df['timestamps'], df['volume'], width=0.6, color='lightblue', alpha=0.7)
        ax2.set_ylabel('成交量')
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # 格式化x轴
        fig.autofmt_xdate()
        
        plt.tight_layout()
        plt.show()
        
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
    
    # 绘制K线图
    print("\n正在绘制K线图...")
    plot_kline_chart(kline_data, "使用Kronos模型基于真实数据生成的预测K线图")
    
    # 可选：使用matplotlib绘制（如果需要）
    # plot_matplotlib_kline(kline_data, "K线图 (matplotlib)")
    
    # 保存数据到CSV
    csv_filename = f"generated_kline_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    kline_data.to_csv(csv_filename, index=False)
    print(f"\n生成的K线数据已保存到: {csv_filename}")


if __name__ == "__main__":
    main()