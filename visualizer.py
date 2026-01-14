"""
可视化模块
生成K线图、信号图和回测结果图表
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    def __init__(self, historical_data, predictions, signals, backtest_results):
        """
        初始化可视化器
        
        Args:
            historical_data (pd.DataFrame): 历史价格数据
            predictions (pd.DataFrame): 预测结果
            signals (pd.DataFrame): 交易信号
            backtest_results (dict): 回测结果
        """
        self.historical_data = historical_data
        self.predictions = predictions
        self.signals = signals
        self.backtest_results = backtest_results
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def create_prediction_chart(self, output_dir):
        """
        创建预测K线图
        
        Args:
            output_dir (str): 输出目录
        """
        try:
            # 创建交互式K线图
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('价格走势', '成交量'),
                row_width=[0.7, 0.3]
            )
            
            # 添加历史数据K线
            if 'Date' in self.historical_data.columns:
                historical_dates = self.historical_data['Date']
            else:
                historical_dates = self.historical_data.index
            
            # 历史K线
            fig.add_trace(
                go.Candlestick(
                    x=historical_dates,
                    open=self.historical_data['Open'] if 'Open' in self.historical_data.columns else self.historical_data.iloc[:, 1],
                    high=self.historical_data['High'] if 'High' in self.historical_data.columns else self.historical_data.iloc[:, 2],
                    low=self.historical_data['Low'] if 'Low' in self.historical_data.columns else self.historical_data.iloc[:, 3],
                    close=self.historical_data['Close'] if 'Close' in self.historical_data.columns else self.historical_data.iloc[:, 4],
                    name='历史价格'
                ),
                row=1, col=1
            )
            
            # 预测K线（使用虚线表示预测）
            if not self.predictions.empty and 'Date' in self.predictions.columns:
                fig.add_trace(
                    go.Candlestick(
                        x=self.predictions['Date'],
                        open=self.predictions['Predicted_Open'],
                        high=self.predictions['Predicted_High'],
                        low=self.predictions['Predicted_Low'],
                        close=self.predictions['Predicted_Close'],
                        name='预测价格',
                        line=dict(width=1)  # 修复：使用width而不是color
                    ),
                    row=1, col=1
                )
            
            # 添加历史成交量
            if 'Volume' in self.historical_data.columns:
                fig.add_trace(
                    go.Bar(
                        x=historical_dates,
                        y=self.historical_data['Volume'],
                        name='历史成交量',
                        marker_color='lightblue',
                        showlegend=True
                    ),
                    row=2, col=1
                )
            
            # 添加预测成交量
            if not self.predictions.empty and 'Predicted_Volume' in self.predictions.columns:
                fig.add_trace(
                    go.Bar(
                        x=self.predictions['Date'],
                        y=self.predictions['Predicted_Volume'],
                        name='预测成交量',
                        marker_color='orange',
                        showlegend=True
                    ),
                    row=2, col=1
                )
            
            # 更新布局
            fig.update_layout(
                title='恒指历史与预测K线图',
                yaxis_title='价格',
                xaxis_rangeslider_visible=False,
                height=800,
                width=1200
            )
            
            # 保存图表
            output_path = os.path.join(output_dir, f"prediction_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            fig.write_html(output_path)
            logger.info(f"预测K线图已保存至: {output_path}")
            
            # 同时保存静态图片
            fig.write_image(os.path.join(output_dir, f"prediction_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
            
        except Exception as e:
            logger.error(f"创建预测K线图时出错: {str(e)}")
    
    def create_signals_chart(self, output_dir):
        """
        创建交易信号图表
        
        Args:
            output_dir (str): 输出目录
        """
        try:
            if self.signals.empty:
                logger.info("没有交易信号，跳过信号图表生成")
                return
            
            # 创建价格图表
            fig = go.Figure()
            
            # 添加收盘价线
            if 'Date' in self.historical_data.columns and 'Close' in self.historical_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.historical_data['Date'],
                        y=self.historical_data['Close'],
                        mode='lines',
                        name='实际收盘价',
                        line=dict(color='blue')
                    )
                )
            else:
                # 如果没有日期列，则使用索引
                fig.add_trace(
                    go.Scatter(
                        x=self.historical_data.index,
                        y=self.historical_data.iloc[:, -1] if len(self.historical_data.columns) > 0 else self.historical_data,
                        mode='lines',
                        name='实际收盘价',
                        line=dict(color='blue')
                    )
                )
            
            # 添加预测收盘价
            if not self.predictions.empty and 'Date' in self.predictions.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.predictions['Date'],
                        y=self.predictions['Predicted_Close'],
                        mode='lines',
                        name='预测收盘价',
                        line=dict(color='red', dash='dash')
                    )
                )
            
            # 标记买入信号
            buy_signals = self.signals[self.signals['Signal'] == 'BUY']
            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals['Date'],
                        y=[self.historical_data['Close'].max() * 0.95] * len(buy_signals),  # 在价格下方标记
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=15, color='green'),
                        name='买入信号',
                        text=buy_signals['Reason'],
                        hovertemplate='<b>买入信号</b><br>%{text}<extra></extra>'
                    )
                )
            
            # 标记卖出信号
            sell_signals = self.signals[self.signals['Signal'] == 'SELL']
            if not sell_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals['Date'],
                        y=[self.historical_data['Close'].max() * 0.95] * len(sell_signals),  # 在价格下方标记
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=15, color='red'),
                        name='卖出信号',
                        text=sell_signals['Reason'],
                        hovertemplate='<b>卖出信号</b><br>%{text}<extra></extra>'
                    )
                )
            
            fig.update_layout(
                title='交易信号图表',
                xaxis_title='日期',
                yaxis_title='价格',
                height=600,
                width=1200
            )
            
            # 保存图表
            output_path = os.path.join(output_dir, f"signals_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            fig.write_html(output_path)
            logger.info(f"交易信号图表已保存至: {output_path}")
            
            # 同时保存静态图片
            fig.write_image(os.path.join(output_dir, f"signals_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
            
        except Exception as e:
            logger.error(f"创建交易信号图表时出错: {str(e)}")
    
    def create_backtest_summary(self, output_dir):
        """
        创建回测结果摘要和统计
        
        Args:
            output_dir (str): 输出目录
        """
        try:
            # 创建结果摘要图表
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    '资产价值变化', 
                    '收益分布', 
                    '回测统计指标', 
                    '交易详情'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"type": "table"}, {"type": "table"}]]
            )
            
            # 1. 资产价值变化（如果有交易记录）
            if not self.backtest_results['trades'].empty:
                trades_df = self.backtest_results['trades']
                fig.add_trace(
                    go.Scatter(
                        x=trades_df['Date'],
                        y=trades_df['Total_Value'],
                        mode='lines+markers',
                        name='总资产价值',
                        line=dict(color='green')
                    ),
                    row=1, col=1
                )
            else:
                # 如果没有交易，显示初始资金线
                fig.add_hline(
                    y=self.backtest_results['initial_capital'],
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="初始资金",
                    row=1, col=1
                )
            
            # 2. 收益统计表格
            stats_data = [
                ['指标', '数值'],
                ['初始资金', f"¥{self.backtest_results['initial_capital']:,.2f}"],
                ['最终资金', f"¥{self.backtest_results['final_capital']:,.2f}"],
                ['总收益', f"¥{self.backtest_results['total_profit']:,.2f}"],
                ['总收益率', f"{self.backtest_results['total_return']:.2%}"],
                ['交易次数', f"{self.backtest_results['num_trades']}"],
                ['信号使用数', f"{self.backtest_results['signals_used']}"],
                ['胜率', f"{self.backtest_results['win_rate']:.2%}"],
                ['最大回撤', f"{self.backtest_results['max_drawdown']:.2%}"]
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=stats_data[0]),
                    cells=dict(values=list(map(list, zip(*stats_data[1:])))),
                    name='回测统计'
                ),
                row=2, col=1
            )
            
            # 3. 交易详情表格
            if not self.backtest_results['trades'].empty:
                trades_df = self.backtest_results['trades']
                trade_details = [
                    ['日期', '类型', '价格', '股数', '金额', '现金余额']
                ]
                
                for _, row in trades_df.iterrows():
                    if row['Type'] == 'BUY':
                        trade_details.append([
                            row['Date'],
                            row['Type'],
                            f"{row['Price']:.2f}",
                            f"{row['Shares']:,}",
                            f"-{row['Cost']:.2f}",
                            f"{row['Cash']:.2f}"
                        ])
                    else:  # SELL
                        trade_details.append([
                            row['Date'],
                            row['Type'],
                            f"{row['Price']:.2f}",
                            f"{row['Shares']:,}",
                            f"+{row['Revenue']:.2f}",
                            f"{row['Cash']:.2f}"
                        ])
                
                fig.add_trace(
                    go.Table(
                        header=dict(values=trade_details[0]),
                        cells=dict(values=list(map(list, zip(*trade_details[1:])))),
                        name='交易详情'
                    ),
                    row=2, col=2
                )
            else:
                # 没有交易时的提示
                no_trade_msg = [
                    ['信息'],
                    ['无交易记录']
                ]
                fig.add_trace(
                    go.Table(
                        header=dict(values=no_trade_msg[0]),
                        cells=dict(values=list(map(list, zip(*no_trade_msg[1:])))),
                        name='无交易'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title='回测结果摘要',
                height=800,
                width=1400
            )
            
            # 保存图表
            output_path = os.path.join(output_dir, f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            fig.write_html(output_path)
            logger.info(f"回测摘要已保存至: {output_path}")
            
            # 同时保存静态图片
            fig.write_image(os.path.join(output_dir, f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
            
            # 保存详细报告为CSV
            self._save_detailed_report(output_dir)
            
        except Exception as e:
            logger.error(f"创建回测摘要时出错: {str(e)}")
    
    def _save_detailed_report(self, output_dir):
        """
        保存详细的回测报告
        
        Args:
            output_dir (str): 输出目录
        """
        try:
            report_path = os.path.join(output_dir, f"detailed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            
            # 创建综合报告
            report_data = {
                'Metric': [
                    'Initial Capital',
                    'Final Capital', 
                    'Total Profit',
                    'Total Return (%)',
                    'Number of Trades',
                    'Signals Used',
                    'Win Rate (%)',
                    'Max Drawdown (%)',
                    'Run Date'
                ],
                'Value': [
                    self.backtest_results['initial_capital'],
                    self.backtest_results['final_capital'],
                    self.backtest_results['total_profit'],
                    self.backtest_results['total_return'] * 100,
                    self.backtest_results['num_trades'],
                    self.backtest_results['signals_used'],
                    self.backtest_results['win_rate'] * 100,
                    self.backtest_results['max_drawdown'] * 100,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            
            report_df = pd.DataFrame(report_data)
            report_df.to_csv(report_path, index=False)
            logger.info(f"详细报告已保存至: {report_path}")
            
            # 如果有交易记录，也保存交易详情
            if not self.backtest_results['trades'].empty:
                trades_path = os.path.join(output_dir, f"trades_detail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                self.backtest_results['trades'].to_csv(trades_path, index=False)
                logger.info(f"交易详情已保存至: {trades_path}")
                
        except Exception as e:
            logger.error(f"保存详细报告时出错: {str(e)}")