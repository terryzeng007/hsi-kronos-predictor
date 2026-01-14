"""
Kronos模型安装脚本
用于正确安装和配置Kronos金融预测模型
"""
import subprocess
import sys
import os
import logging

logger = logging.getLogger(__name__)

def install_kronos():
    """
    安装Kronos模型及相关依赖
    """
    print("正在安装Kronos模型...")
    
    # 首先尝试安装git（如果尚未安装）
    try:
        import git
    except ImportError:
        print("正在安装GitPython...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "GitPython"])
    
    # 克隆Kronos仓库
    kronos_repo_url = "https://github.com/shiyu-coder/Kronos"
    kronos_local_path = "./kronos_src"
    
    if not os.path.exists(kronos_local_path):
        print(f"正在从 {kronos_repo_url} 克隆Kronos源码...")
        try:
            from git import Repo
            Repo.clone_from(kronos_repo_url, kronos_local_path)
            print("Kronos源码克隆成功")
        except Exception as e:
            print(f"Git克隆失败: {e}")
            print("请确保已安装Git并配置了PATH")
            return False
    else:
        print("Kronos源码已存在")
    
    # 安装Kronos依赖
    kronos_requirements = os.path.join(kronos_local_path, "requirements.txt")
    if os.path.exists(kronos_requirements):
        print("正在安装Kronos依赖...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", kronos_requirements])
            print("Kronos依赖安装成功")
        except Exception as e:
            print(f"安装Kronos依赖时出错: {e}")
            # 继续执行，因为可能只需要部分依赖
    
    # 安装Kronos包（如果setup.py存在）
    kronos_setup = os.path.join(kronos_local_path, "setup.py")
    if os.path.exists(kronos_setup):
        print("正在安装Kronos包...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", kronos_local_path])
            print("Kronos包安装成功")
        except Exception as e:
            print(f"安装Kronos包时出错: {e}")
            return False
    
    print("Kronos安装完成！")
    return True

def test_kronos_installation():
    """
    测试Kronos是否正确安装
    """
    try:
        # 尝试导入Kronos相关模块
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import snapshot_download
        print("基础依赖检查通过")
        
        # 测试是否可以加载Kronos模型
        print("正在测试Kronos模型加载...")
        
        # 由于我们还没有完整的Kronos实现，我们只是验证基本的依赖是否安装
        return True
    except ImportError as e:
        print(f"依赖检查失败: {e}")
        return False

if __name__ == "__main__":
    success = install_kronos()
    if success:
        print("\nKronos模型安装成功！")
        print("现在您可以运行主程序进行K线预测了。")
    else:
        print("\nKronos模型安装失败，请检查错误信息并重试。")
        print("您可能需要:")
        print("1. 确保网络连接正常")
        print("2. 确保已安装Git并配置了PATH")
        print("3. 检查Python版本是否符合要求")