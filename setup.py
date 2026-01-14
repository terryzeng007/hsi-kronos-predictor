#!/usr/bin/env python
"""
项目安装脚本
用于安装依赖项并准备运行环境
"""
import subprocess
import sys
import os
from pathlib import Path


def install_dependencies():
    """安装项目依赖"""
    print("正在安装项目依赖...")
    
    try:
        # 升级pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # 安装requirements.txt中的依赖
        req_file = Path("requirements.txt")
        if req_file.exists():
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
            print("✅ 依赖安装完成")
        else:
            print("⚠️ 未找到requirements.txt文件")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装依赖时出错: {e}")
        sys.exit(1)


def create_directories():
    """创建必要的目录"""
    print("正在创建项目目录...")
    
    directories = [
        "data",  # 存放原始数据
        "models",  # 存放模型文件
        "predict_price",  # 存放预测价格
        "predict_graph",  # 存放预测图表
        "predict_signal",  # 存放交易信号
        "backtest_result"  # 存放回测结果
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 已创建目录: {directory}")


def check_cuda():
    """检查CUDA是否可用"""
    print("\n正在检查CUDA支持...")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
        
        if cuda_available:
            print(f"✅ CUDA可用，GPU数量: {gpu_count}")
            print(f"GPU型号: {torch.cuda.get_device_name(0) if gpu_count > 0 else 'N/A'}")
        else:
            print("⚠️ CUDA不可用，将使用CPU进行计算")
            
        # 设置环境变量
        os.environ["CUDA_AVAILABLE"] = str(cuda_available).lower()
        
    except ImportError:
        print("⚠️ 未安装PyTorch，CUDA检查跳过")


def setup_project():
    """完整的项目设置流程"""
    print("🚀 开始设置恒指K线预测项目...")
    print(f"项目路径: {os.getcwd()}")
    
    # 创建目录
    create_directories()
    
    # 检查CUDA
    check_cuda()
    
    # 安装依赖
    install_dependencies()
    
    print("\n✅ 项目设置完成！")
    print("\n📋 接下来您可以：")
    print("   1. 确保数据文件位于 D:/Git_Project/data/HSI.xlsx")
    print("   2. 运行 'python main.py' 开始预测任务")
    print("   3. 查看生成的预测结果和图表")


if __name__ == "__main__":
    setup_project()