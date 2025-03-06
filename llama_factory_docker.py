import subprocess
import sys
import os

def check_docker_installed():
    try:
        subprocess.run(["docker", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Docker已安裝。")
        return True
    except subprocess.CalledProcessError:
        print("未检测到Docker安裝。")
        return False

def install_docker():
    print("自動安裝Docker...")
    if sys.platform.startswith('linux'):
        subprocess.run(["sudo", "apt-get", "update"])
        subprocess.run(["sudo", "apt-get", "install", "-y", "docker-ce", "docker-ce-cli", "containerd.io"])
    elif sys.platform.startswith('win'):
        print("請手動安裝Docker Desktop並啟用Hyper-V。")
        sys.exit(1)
    else:
        print("不支援的作業系統。")
        sys.exit(1)

def check_existing_containers():
    try:
        result = subprocess.run(["docker", "ps", "-a", "--format", "{{.ID}}\t{{.Names}}"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        containers = result.stdout.decode().strip().split('\n')
        llama_containers = [c for c in containers if "llama-factory" in c]
        if len(llama_containers) > 0:
            print("已找到現有的llama-factory容器：")
            for i, container in enumerate(llama_containers):
                print(f"{i+1}. {container}")
            choice = input("選擇容器（輸入編號）或輸入n新建容器：")
            if choice.lower() == 'n':
                return None
            else:
                return llama_containers[int(choice)-1].split('\t')[0]
        else:
            return None
    except subprocess.CalledProcessError:
        print("無法獲取現有容器。")
        return None

def create_docker_container():
    print("創建新的Docker容器...")
    try:
        subprocess.run(["docker", "pull", "llama-factory"], check=True)
        subprocess.run(["docker", "run", "-d", "--name", "llama-factory-container", "-v", "/path/to/data:/data", "llama-factory"], check=True)
        return "llama-factory-container"
    except subprocess.CalledProcessError:
        print("創建容器失敗。")
        sys.exit(1)

def check_nvidia_gpu():
    try:
        subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("已檢測到NVIDIA GPU。")
        return True
    except subprocess.CalledProcessError:
        print("未檢測到NVIDIA GPU。")
        return False

def configure_nvidia_docker():
    print("配置NVIDIA Container Toolkit...")
    if sys.platform.startswith('linux'):
        subprocess.run(["sudo", "apt-get", "install", "-y", "nvidia-container-toolkit"])
        subprocess.run(["sudo", "nvidia-ctk", "runtime", "configure", "--runtime=docker"])
        subprocess.run(["sudo", "systemctl", "restart", "docker"])
    else:
        print("NVIDIA Container Toolkit在Windows上不支援。")

def install_llama_factory(container_id):
    print("安裝llama factory及其依賴...")
    try:
        subprocess.run(["docker", "exec", "-it", container_id, "pip", "install", "-r", "/data/requirements.txt"], check=True)
    except subprocess.CalledProcessError:
        print("安裝llama factory失敗。")
        sys.exit(1)

def start_web_ui(container_id):
    print("啟動llama factory Web UI...")
    try:
        subprocess.run(["docker", "exec", "-it", container_id, "python", "/data/run_webui.py"], check=True)
    except subprocess.CalledProcessError:
        print("啟動Web UI失敗。")
        sys.exit(1)

def main():
    if not check_docker_installed():
        install_choice = input("是否要自動安裝Docker？(y/n): ")
        if install_choice.lower() == 'y':
            install_docker()
        else:
            print("請手動安裝Docker後再運行此程式。")
            sys.exit(1)

    existing_container = check_existing_containers()
    if existing_container:
        container_id = existing_container
    else:
        container_id = create_docker_container()

    if check_nvidia_gpu():
        configure_nvidia_docker()

    install_llama_factory(container_id)
    start_web_ui(container_id)

if __name__ == "__main__":
    main()