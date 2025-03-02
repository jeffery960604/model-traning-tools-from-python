import os
import subprocess
import importlib
import webbrowser

# 定义所需依赖项和目標網站
REQUIRED_PACKAGES = [
    "torch",
    "gradio",
    "llama-factory",
    "bitsandbytes",
    "modelscope",
    "vllm",
]
REDIRECT_URL = "https://github.com/hiyouga/LLaMA-Factory"  # 替换为你的目标网站

def install_package(package):
    """安装指定的 Python 包"""
    try:
        subprocess.check_call([os.sys.executable, "-m", "pip", "install", package])
        print(f"已成功安装 {package}")
    except subprocess.CalledProcessError:
        print(f"安装 {package} 失敗")
        os.sys.exit(1)

def check_and_install_dependencies():
    """檢查并安装依赖项"""
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"已安装 {package}")
        except ImportError:
            print(f"未检测到 {package}，正在安装...")
            install_package(package)

def launch_web_ui():
    """启动 Web UI"""
    try:
        # 替换为启动 LLaMA-Factory Web UI 的命令
        subprocess.Popen(["llamafactory-cli", "webui"])
        print("LLaMA-Factory Web UI 已启动")
    except FileNotFoundError:
        print("启动 LLaMA-Factory Web UI 失败，请确保已正确安装")
        os.sys.exit(1)

def main():
    """主程序"""
    print("正在检查依赖项...")
    check_and_install_dependencies()
    
    print("正在启动 Web UI...")
    launch_web_ui()
    
    print(f"正在引导您到指定網站 {REDIRECT_URL}")
    webbrowser.open(REDIRECT_URL)

if __name__ == "__main__":
    main()