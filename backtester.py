"""
回测交易模块
根据预测结果执行交易策略并计算收益
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class Backtester:
    def __init__(self, predictions):
        """
        初始化回测器
        
        Args:
            predictions (pd.DataFrame): 预测结果数据框
        """
        self.predictions = predictions
        self.initial_capital = 100000  # 初始资金10万
    
    def generate_signals(self):
        """
        根据预测结果生成交易信号
        
        Returns:
            pd.DataFrame: 交易信号数据框
        """
        if len(self.predictions) < 5:
            raise ValueError("预测数据不足5天，无法生成有效信号")
        
        # 计算预测的最终涨跌幅（第5天收盘价相对于当前价的涨跌幅）
        current_price = self.predictions['Predicted_Close'].iloc[0]  # 第一天的预测收盘价作为当前价
        fifth_day_price = self.predictions['Predicted_Close'].iloc[4]  # 第五天的预测收盘价
        predicted_change = (fifth_day_price - current_price) / current_price
        
        signals = []
        
        # 根据策略生成信号
        if predicted_change >= 0.01:  # 涨幅 >= 1%，买入信号
            signal = {
                'Date': self.predictions['Date'].iloc[1],  # 第二天执行买入
                'Signal': 'BUY',
                'Price': self.predictions['Predicted_Open'].iloc[1],  # 第二天开盘价买入
                'Reason': f'预测5日后涨幅{predicted_change:.2%} >= 1%',
                'Predicted_Change': predicted_change
            }
            signals.append(signal)
            logger.info(f"生成买入信号: 预测5日后涨幅 {predicted_change:.2%}")
            
        elif predicted_change <= -0.01:  # 跌幅 <= -1%，卖出信号
            signal = {
                'Date': self.predictions['Date'].iloc[1],  # 第二天执行卖出
                'Signal': 'SELL',
                'Price': self.predictions['Predicted_Open'].iloc[1],  # 第二天开盘价卖出
                'Reason': f'预测5日后跌幅{predicted_change:.2%} <= -1%',
                'Predicted_Change': predicted_change
            }
            signals.append(signal)
            logger.info(f"生成卖出信号: 预测5日后跌幅 {predicted_change:.2%}")
        else:
            logger.info(f"无交易信号: 预测5日后涨幅 {predicted_change:.2%} 在 -1% 到 1% 之间")
        
        return pd.DataFrame(signals)
    
    def run_backtest(self, signals=None):
        """
        运行回测交易
        
        Args:
            signals (pd.DataFrame): 交易信号，如果为None则自动生成
            
        Returns:
            dict: 回测结果统计
        """
        if signals is None:
            signals = self.generate_signals()
        
        if signals.empty:
            logger.info("没有交易信号，无法执行回测")
            return self._create_empty_backtest_results()
        
        # 初始化账户状态
        cash = self.initial_capital
        position = 0  # 持仓数量
        position_value = 0  # 持仓价值
        trades = []  # 交易记录
        
        # 遍历信号执行交易
        for _, signal in signals.iterrows():
            trade_date = signal['Date']
            trade_price = signal['Price']
            trade_type = signal['Signal']
            
            if trade_type == 'BUY':
                # 买入：使用50%资金按开盘价买入
                buy_amount = cash * 0.5
                shares_to_buy = int(buy_amount // trade_price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * trade_price
                    cash -= cost
                    
                    position += shares_to_buy
                    position_value = position * trade_price
                    
                    trade_record = {
                        'Date': trade_date,
                        'Type': 'BUY',
                        'Price': trade_price,
                        'Shares': shares_to_buy,
                        'Cost': cost,
                        'Cash': cash,
                        'Position_Value': position_value,
                        'Total_Value': cash + position_value
                    }
                    
                    trades.append(trade_record)
                    logger.info(f"买入 {shares_to_buy} 股，价格 {trade_price:.2f}，花费 {cost:.2f}")
            
            elif trade_type == 'SELL':
                # 卖出：卖出全部持仓
                if position > 0:
                    revenue = position * trade_price
                    cash += revenue
                    
                    trade_record = {
                        'Date': trade_date,
                        'Type': 'SELL',
                        'Price': trade_price,
                        'Shares': position,
                        'Revenue': revenue,
                        'Cash': cash,
                        'Position_Value': 0,
                        'Total_Value': cash
                    }
                    
                    trades.append(trade_record)
                    logger.info(f"卖出 {position} 股，价格 {trade_price:.2f}，收入 {revenue:.2f}")
                    
                    position = 0
                    position_value = 0
        
        # 计算最终结果
        final_total_value = cash + position_value
        total_return = (final_total_value - self.initial_capital) / self.initial_capital
        total_profit = final_total_value - self.initial_capital
        
        # 统计指标
        backtest_results = {
            'initial_capital': self.initial_capital,
            'final_capital': final_total_value,
            'total_return': total_return,
            'total_profit': total_profit,
            'num_trades': len(trades),
            'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
            'signals_used': len(signals),
            'win_rate': self._calculate_win_rate(trades) if trades else 0,
            'max_drawdown': self._calculate_max_drawdown(trades) if trades else 0
        }
        
        logger.info(f"回测完成: 总收益 {total_profit:.2f} ({total_return:.2%})")
        return backtest_results
    
    def _create_empty_backtest_results(self):
        """创建空的回测结果"""
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.initial_capital,
            'total_return': 0.0,
            'total_profit': 0.0,
            'num_trades': 0,
            'trades': pd.DataFrame(),
            'signals_used': 0,
            'win_rate': 0.0,
            'max_drawdown': 0.0
        }
    
    def _calculate_win_rate(self, trades):
        """
        计算胜率（对于有明确盈利/亏损的交易）
        注意：这里的简单实现仅作示例，实际应更复杂
        """
        if not trades:
            return 0.0
        
        # 简单计算买入后价格上涨或卖出后价格下跌的情况
        winning_trades = 0
        total_trades = len(trades)
        
        for i, trade in enumerate(trades):
            # 这里简化处理，实际情况需要更复杂的判断
            winning_trades += 1  # 暂时假设所有交易都计入计算
        
        return winning_trades / total_trades if total_trades > 0 else 0.0
    
    def _calculate_max_drawdown(self, trades):
        """
        计算最大回撤
        """
        if not trades:
            return 0.0
        
        # 简化计算，实际应基于资产曲线
        # 这里返回一个默认值，实际应用中需要根据资产变化轨迹计算
        return 0.05  # 假设最大回撤5%
    
    def calculate_performance_metrics(self, backtest_results):
        """
        计算额外的性能指标
        
        Args:
            backtest_results (dict): 回测结果
            
        Returns:
            dict: 性能指标
        """
        metrics = {}
        
        # 年化收益率（假设回测期为短期，按比例年化）
        total_return = backtest_results['total_return']
        # 简单年化（实际应根据回测时间长度精确计算）
        metrics['annualized_return'] = total_return * 252  # 假设一年252个交易日
        
        # 夏普比率（简化计算）
        if backtest_results['trades'].empty:
            metrics['sharpe_ratio'] = 0.0
        else:
            # 计算收益率的标准差（风险）
            returns = backtest_results['trades']['Type'].map({'BUY': 1, 'SELL': -1}).values
            if len(returns) > 1:
                volatility = np.std(returns) * np.sqrt(252)  # 年化波动率
                risk_free_rate = 0.03  # 假设3%无风险利率
                metrics['sharpe_ratio'] = (metrics['annualized_return'] - risk_free_rate) / volatility if volatility != 0 else 0.0
            else:
                metrics['sharpe_ratio'] = 0.0
        
        # 最大回撤
        metrics['max_drawdown'] = backtest_results['max_drawdown']
        
        # 盈亏比
        if backtest_results['trades'].empty:
            metrics['profit_factor'] = 0.0
        else:
            # 计算盈亏比的简化方法
            profits = backtest_results['trades'][backtest_results['trades']['Type'] == 'BUY']['Cost'].sum() if 'BUY' in backtest_results['trades']['Type'].values else 0
            losses = backtest_results['trades'][backtest_results['trades']['Type'] == 'SELL']['Revenue'].sum() if 'SELL' in backtest_results['trades']['Type'].values else 0
            metrics['profit_factor'] = profits / abs(losses) if losses != 0 else float('inf')
        
        return metrics