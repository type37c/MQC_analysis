"""
発見した量子データセットを実際にダウンロード
"""
import requests
import os
import zipfile
import json
from known_quantum_datasets import KNOWN_QUANTUM_DATASETS, get_direct_download_links

def download_dataset(url, save_path):
    """データセットをダウンロード"""
    
    print(f"ダウンロード中: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # ファイルサイズを取得
        total_size = int(response.headers.get('content-length', 0))
        
        # ダウンロード
        with open(save_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # 進捗表示
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"  進捗: {percent:.1f}%", end='\r')
        
        print(f"\n  → 保存完了: {save_path}")
        return True
        
    except Exception as e:
        print(f"  → エラー: {e}")
        return False

def download_github_data(github_url, save_dir):
    """GitHubリポジトリからデータを取得"""
    
    # リポジトリ名を抽出
    parts = github_url.rstrip('/').split('/')
    owner = parts[-2]
    repo = parts[-1]
    
    # GitHubのZIPダウンロードURL
    zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip"
    
    zip_path = os.path.join(save_dir, f"{repo}.zip")
    
    if download_dataset(zip_url, zip_path):
        # ZIPを解凍
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            extract_dir = os.path.join(save_dir, repo)
            zip_ref.extractall(extract_dir)
            print(f"  → 解凍完了: {extract_dir}")
        
        # データファイルを探す
        find_data_files(extract_dir)

def find_data_files(directory):
    """ディレクトリ内のデータファイルを探す"""
    
    data_extensions = ['.csv', '.json', '.h5', '.hdf5', '.npz', '.pkl', '.dat']
    found_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in data_extensions):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                
                if file_size > 1:  # 1KB以上のファイル
                    found_files.append({
                        'path': file_path,
                        'name': file,
                        'size_kb': file_size
                    })
    
    if found_files:
        print(f"\n  発見したデータファイル:")
        for f in found_files[:10]:  # 最初の10個を表示
            print(f"    - {f['name']} ({f['size_kb']:.1f} KB)")
    
    return found_files

def main():
    """メイン実行関数"""
    
    print("=== 量子実験データのダウンロード ===\n")
    
    # 保存ディレクトリの作成
    base_dir = 'downloaded_quantum_data'
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. 既知のデータセットをダウンロード
    print("1. 既知の量子データセットをダウンロード")
    direct_links = get_direct_download_links()
    
    for link in direct_links:
        print(f"\n【{link['name']}】")
        
        if link['type'] == 'github':
            save_dir = os.path.join(base_dir, link['name'].replace(' ', '_'))
            os.makedirs(save_dir, exist_ok=True)
            download_github_data(link['url'], save_dir)
        
        elif link['type'] == 'direct':
            filename = link['url'].split('/')[-1]
            save_path = os.path.join(base_dir, filename)
            download_dataset(link['url'], save_path)
    
    # 2. CSVファイルから追加のソースをダウンロード
    if os.path.exists('arxiv_quantum_data/downloadable_sources.csv'):
        print("\n\n2. arXiv検索で見つかったデータソース")
        import pandas as pd
        
        sources_df = pd.read_csv('arxiv_quantum_data/downloadable_sources.csv')
        
        for _, row in sources_df.head(5).iterrows():  # 最初の5個
            print(f"\n【{row['title'][:50]}...】")
            
            if row['data_source'] == 'github':
                save_dir = os.path.join(base_dir, f"arxiv_{row['arxiv_id']}")
                os.makedirs(save_dir, exist_ok=True)
                download_github_data(row['url'], save_dir)
    
    print("\n=== ダウンロード完了 ===")
    print(f"データは {base_dir} に保存されました")

if __name__ == "__main__":
    main()