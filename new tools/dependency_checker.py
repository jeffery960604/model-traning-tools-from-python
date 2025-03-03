import subprocess
import sys
import platform
from importlib.metadata import version, PackageNotFoundError

REQUIREMENTS = {
    'python': {
        'torch': '2.0.0',
        'transformers': '4.30.0',
        'llama-cpp-python': '0.2.23',
        'flask': '2.3.0',
        'safetensors': '0.3.3'
    },
    'system': {
        'linux': ['cmake', 'build-essential', 'python3-dev'],
        'darwin': ['cmake', 'protobuf-compiler']
    }
}

def check_system_deps():
    system = platform.system().lower()
    if system not in ['linux', 'darwin']:
        raise OSError("Unsupported operating system")
    
    missing = []
    for pkg in REQUIREMENTS['system'].get(system, []):
        try:
            subprocess.check_output(['which', pkg], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            missing.append(pkg)
    
    if missing:
        install_cmd = 'apt-get install' if system == 'linux' else 'brew install'
        print(f"Missing system packages: {', '.join(missing)}")
        subprocess.run(f'sudo {install_cmd} {" ".join(missing)}', shell=True)

def check_python_deps():
    missing = []
    outdated = []
    
    for pkg, req_version in REQUIREMENTS['python'].items():
        try:
            installed_version = version(pkg)
            if installed_version < req_version:
                outdated.append(f"{pkg}>={req_version}")
        except PackageNotFoundError:
            missing.append(pkg)
    
    if missing or outdated:
        install_cmd = ['pip', 'install', '-U']
        install_cmd += missing
        install_cmd += outdated
        subprocess.check_call(install_cmd)

if __name__ == "__main__":
    check_system_deps()
    check_python_deps()