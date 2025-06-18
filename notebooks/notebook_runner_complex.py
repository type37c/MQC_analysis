"""
複素演算CQT解析ノートブック実行ヘルパー
"""
import subprocess
import sys
import os

def install_jupyter_if_needed():
    """Install Jupyter if not available"""
    try:
        import jupyter
        print("✓ Jupyter already installed")
    except ImportError:
        print("Installing Jupyter...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'jupyter'])
        print("✓ Jupyter installed successfully")

def install_required_packages():
    """必要なパッケージをインストール"""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'scipy', 'scikit-learn', 'jupyter'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', package])
            print(f"✓ {package} installed successfully")

def run_notebook_server():
    """Start Jupyter notebook server"""
    print("複素演算CQT解析ノートブックサーバーを起動中...")
    print("ノートブック場所: notebooks/04_complex_cqt_deep_analysis.ipynb")
    print("")
    print("ノートブックを開くには:")
    print("1. ウェブブラウザを開く")
    print("2. http://localhost:8888 にアクセス")
    print("3. notebooks/04_complex_cqt_deep_analysis.ipynb を開く")
    print("")
    print("サーバーを停止するには Ctrl+C を押してください")
    print("-" * 60)
    
    # Change to project root directory
    os.chdir('/home/type37c/projects/CQT_experiments')
    
    # Start Jupyter
    subprocess.run(['jupyter', 'notebook', '--ip=0.0.0.0', '--port=8888', '--no-browser', '--allow-root'])

def check_data_availability():
    """データファイルの存在確認"""
    print("データファイルの確認中...")
    
    # 必要なデータファイル
    required_files = [
        'data_collection/collected_data/bell_states/bell_measurement_data.csv',
        'data_collection/downloaded_quantum_data/IBM_Quantum_Volume/qiskit-experiments/qiskit-experiments-main/test/library/quantum_volume/qv_data_high_noise.json',
        'src/complex_cqt_operations.py',
        'src/complex_error_detection.py',
        'src/cqt_tracker_v3.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    if missing_files:
        print("\n⚠ 以下のファイルが見つかりません:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\n必要に応じて以下のコマンドを実行してください:")
        print("  cd data_collection && python3 run_collection.py")
        print("  cd data_collection && python3 download_quantum_datasets.py")
        print("  cd src && python3 run_complex_cqt_analysis.py")
    else:
        print("\n✓ 全ての必要ファイルが利用可能です")

if __name__ == "__main__":
    print("=== 複素演算CQT解析ノートブック起動準備 ===")
    
    # パッケージインストール確認
    print("\n1. 必要パッケージの確認...")
    install_required_packages()
    
    # Jupyterインストール確認
    print("\n2. Jupyter環境の確認...")
    install_jupyter_if_needed()
    
    # データファイル確認
    print("\n3. データファイルの確認...")
    check_data_availability()
    
    # ノートブックサーバー起動
    print("\n4. ノートブックサーバー起動...")
    run_notebook_server()