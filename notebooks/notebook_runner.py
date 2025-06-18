"""
Jupyter notebook execution helper for CQT analysis
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

def run_notebook_server():
    """Start Jupyter notebook server"""
    print("Starting Jupyter notebook server...")
    print("Notebook location: notebooks/03_cqt_real_data_analysis.ipynb")
    print("")
    print("To open the notebook:")
    print("1. Open your web browser")
    print("2. Go to: http://localhost:8888")
    print("3. Navigate to: notebooks/03_cqt_real_data_analysis.ipynb")
    print("")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Change to project root directory
    os.chdir('/home/type37c/projects/CQT_experiments')
    
    # Start Jupyter
    subprocess.run(['jupyter', 'notebook', '--ip=0.0.0.0', '--port=8888', '--no-browser', '--allow-root'])

if __name__ == "__main__":
    install_jupyter_if_needed()
    run_notebook_server()