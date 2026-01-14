# 恒指K线预测Agent

本项目是一个基于Kronos大模型的恒生指数K线预测系统，能够自动获取历史数据、预测未来价格走势、生成交易信号并进行回测分析。

## 功能特点

- **数据获取**：从本地Excel文件加载恒指历史日价格数据
- **AI预测**：使用Kronos开源大模型预测未来5天K线数据
- **交易策略**：基于预测结果生成买卖信号
- **回测分析**：模拟交易并统计收益指标
- **可视化**：生成K线图、信号图和回测报告

## 技术架构

- **模型**：NeoQuasar/Kronos-base（金融时序预测大模型）
- **框架**：Python + Transformers + Plotly
- **系统**：Windows 10/11，支持CUDA加速

## 项目结构

```
├── data/                    # 数据目录
│   └── HSI.xlsx            # 恒指历史数据
├── predict_price/          # 预测价格数据输出
├── predict_graph/          # 预测K线图输出
├── predict_signal/         # 交易信号输出
├── backtest_result/        # 回测结果输出
├── kronos_src/            # Kronos模型源码（运行setup后自动下载）
├── main.py                # 主程序入口
├── data_loader.py         # 数据加载模块
├── predictor.py           # 预测模块
├── backtester.py          # 回测模块
├── visualizer.py          # 可视化模块
├── kronos_loader.py       # Kronos模型加载器
├── setup_kronos.py        # Kronos模型安装脚本
├── config.py              # 配置文件
├── requirements.txt       # 依赖包列表
└── README.md              # 项目说明
```

## 安装与运行

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 安装Kronos模型

```bash
python setup_kronos.py
```

### 3. 准备数据

将恒指历史数据保存为 `data/HSI.xlsx`，包含以下列：
- Date: 日期
- Open: 开盘价
- High: 最高价
- Low: 最低价
- Close: 收盘价
- Volume: 成交量

### 4. 运行预测

```bash
python main.py
```

## 交易策略
'''运行另外两个0-prediction_rollIng进行滚动预测 之后0-kronos_predict 进行收益率回测'''

-案例预测两周数据
- 当预测未来10天最后涨幅 ≥ 9% 时，生成买入信号
- 当预测未来10天最后跌幅 ≤ -9% 时，生成卖出信号
- 其余情况下不产生交易信号

## 输出结果

- **预测价格**：保存到 `predict_price/` 目录
- **K线图表**：保存到 `predict_graph/` 目录
- **交易信号**：保存到 `predict_signal/` 目录
- **回测报告**：保存到 `backtest_result/` 目录

## 注意事项

- 本项目仅供学习和研究使用，不构成投资建议
- 预测结果仅供参考，实际投资需谨慎
- 请确保数据文件格式正确
- 运行前确保有足够的磁盘空间和内存

## 许可证

MIT License
