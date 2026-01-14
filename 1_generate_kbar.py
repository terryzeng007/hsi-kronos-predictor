"""
使用Kronos模型生成模拟K线数据并绘制K线图
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
    print("警告: 未找到Kronos模型，将使用模拟方法生成K线数据")
    KRONOS_AVAILABLE = False

import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_mock_kline_data(start_date, num_points, initial_price=20000):
    """
    生成模拟K线数据
    
    Args:
        start_date: 开始日期
        num_points: 数据点数量
        initial_price: 初始价格
    
    Returns:
        DataFrame: 包含OHLCV数据的DataFrame
    """
    dates = pd.date_range(start=start_date, periods=num_points, freq='D')
    
    # 生成模拟价格数据
    prices = [initial_price]
    volumes = []
    
    for i in range(1, num_points):
        # 随机价格变动 (-2% 到 +2%)
        change_pct = np.random.normal(0, 0.015)  # 平均无变化，标准差1.5%
        new_price = prices[-1] * (1 + change_pct)
        prices.append(new_price)
        
        # 随机生成成交量
        volume = np.random.randint(500000, 5000000)
        volumes.append(volume)
    
    # 为第一天添加成交量
    volumes.insert(0, np.random.randint(500000, 5000000))
    
    # 生成OHLC数据
    opens = []
    highs = []
    lows = []
    closes = []
    
    for i in range(num_points):
        if i == 0:
            open_price = initial_price
        else:
            open_price = closes[i-1]  # 今天的开盘价是昨天的收盘价
        
        # 添加随机波动来生成高、低价
        close_price = prices[i]
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
    
    df = pd.DataFrame({
        'timestamps': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    return df


def generate_kline_with_kronos(num_points=100):
    """
    使用Kronos模型生成K线数据
    
    Args:
        num_points: 要生成的数据点数量
    
    Returns:
        DataFrame: 生成的K线数据
    """
    if not KRONOS_AVAILABLE:
        print("Kronos模型不可用，使用模拟方法")
        return generate_mock_kline_data(datetime.now() - timedelta(days=num_points), num_points)
    
    try:
        print("正在加载Kronos模型...")
        
        # 加载预训练的Kronos模型和分词器
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")  # 使用small版本以节省资源
        
        # 初始化预测器
        predictor = KronosPredictor(model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # 生成基础历史数据作为输入
        print("生成基础历史数据...")
        historical_data = generate_mock_kline_data(datetime.now() - timedelta(days=200), 200)
        
        # 准备预测所需的数据
        lookback = min(150, len(historical_data))  # 使用最近150个数据点
        pred_len = num_points  # 预测未来的num_points个数据点
        
        x_df = historical_data.tail(lookback)[['open', 'high', 'low', 'close', 'volume']].copy()
        x_timestamp = historical_data.tail(lookback)['timestamps'].reset_index(drop=True)
        
        # 创建未来的时间戳
        last_date = historical_data['timestamps'].iloc[-1]
        y_timestamp = pd.date_range(start=last_date + timedelta(days=1), periods=pred_len, freq='D')
        
        print(f"使用Kronos模型预测未来 {pred_len} 天的数据...")
        
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
        print("回退到模拟方法生成数据")
        return generate_mock_kline_data(datetime.now() - timedelta(days=num_points), num_points)


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
    print("开始生成K线数据...")
    
    # 生成K线数据
    kline_data = generate_kline_with_kronos(num_points=50)  # 生成50个数据点
    
    print(f"生成了 {len(kline_data)} 条K线数据")
    print("数据预览:")
    print(kline_data.head(10))
    
    # 绘制K线图
    print("\n正在绘制K线图...")
    plot_kline_chart(kline_data, "使用Kronos模型生成的K线图")
    
    # 可选：使用matplotlib绘制（如果需要）
    # plot_matplotlib_kline(kline_data, "K线图 (matplotlib)")
    
    # 保存数据到CSV
    csv_filename = f"generated_kline_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    kline_data.to_csv(csv_filename, index=False)
    print(f"\n生成的K线数据已保存到: {csv_filename}")


if __name__ == "__main__":
    main()