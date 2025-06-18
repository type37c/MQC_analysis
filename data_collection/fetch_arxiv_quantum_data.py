"""
arXivから量子計算の実験データを含む論文を検索・取得
"""
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import re
import os
from datetime import datetime
import time

class ArxivQuantumDataFetcher:
    """arXivから量子実験データを取得するクラス"""
    
    def __init__(self, save_dir='arxiv_quantum_data'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.base_url = 'http://export.arxiv.org/api/query'
        
    def search_quantum_datasets(self, max_results=100):
        """量子計算の実験データを含む論文を検索"""
        
        # 検索クエリ（複数の条件をOR検索）
        search_queries = [
            'all:"quantum computing" AND all:"experimental data"',
            'all:"quantum" AND all:"dataset" AND all:"measurement"',
            'all:"bell state" AND all:"experimental results"',
            'all:"quantum supremacy" AND all:"data"',
            'all:"NISQ" AND all:"benchmarks"',
            'all:"quantum processor" AND all:"calibration data"'
        ]
        
        all_papers = []
        
        for query in search_queries:
            print(f"\n検索中: {query}")
            
            params = {
                'search_query': query,
                'start': 0,
                'max_results': max_results // len(search_queries),
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            response = requests.get(self.base_url, params=params)
            time.sleep(3)  # arXivのレート制限対策
            
            if response.status_code == 200:
                papers = self._parse_arxiv_response(response.text)
                all_papers.extend(papers)
                print(f"  → {len(papers)}件の論文を発見")
            
        # 重複を除去
        unique_papers = {p['arxiv_id']: p for p in all_papers}.values()
        
        # データフレームに変換
        df = pd.DataFrame(list(unique_papers))
        df.to_csv(os.path.join(self.save_dir, 'arxiv_papers_with_data.csv'), index=False)
        
        return df
    
    def _parse_arxiv_response(self, xml_content):
        """arXiv APIレスポンスをパース"""
        root = ET.fromstring(xml_content)
        
        # 名前空間の定義
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        papers = []
        
        for entry in root.findall('atom:entry', namespaces):
            # 基本情報を抽出
            paper = {
                'title': entry.find('atom:title', namespaces).text.strip(),
                'arxiv_id': entry.find('atom:id', namespaces).text.split('/')[-1],
                'published': entry.find('atom:published', namespaces).text,
                'summary': entry.find('atom:summary', namespaces).text.strip(),
                'authors': [author.find('atom:name', namespaces).text 
                           for author in entry.findall('atom:author', namespaces)],
                'pdf_link': None,
                'has_data': False,
                'data_keywords': []
            }
            
            # PDFリンクを探す
            for link in entry.findall('atom:link', namespaces):
                if link.get('type') == 'application/pdf':
                    paper['pdf_link'] = link.get('href')
                    break
            
            # データの存在を示すキーワードをチェック
            abstract_lower = paper['summary'].lower()
            data_indicators = [
                'dataset', 'data set', 'experimental data', 'measurement data',
                'raw data', 'supplementary data', 'data available',
                'github.com', 'zenodo.org', 'figshare.com', 'dryad',
                'supplemental material', 'ancillary files'
            ]
            
            found_keywords = []
            for indicator in data_indicators:
                if indicator in abstract_lower:
                    found_keywords.append(indicator)
                    paper['has_data'] = True
            
            paper['data_keywords'] = found_keywords
            
            # GitHubリンクを抽出 (より寛容なパターン)
            github_patterns = [
                r'github\.com/[\w-]+/[\w-]+',
                r'github\.com/[\w\-\.]+/[\w\-\.]+',
                r'https://github\.com/[\w\-\.]+/[\w\-\.]+'
            ]
            
            github_links = []
            for pattern in github_patterns:
                matches = re.findall(pattern, abstract_lower)
                github_links.extend(matches)
            
            # 重複を除去
            github_links = list(set(github_links))
            
            if github_links:
                paper['github_links'] = github_links
            
            papers.append(paper)
        
        return papers
    
    def get_supplementary_data_links(self, arxiv_id):
        """論文の補足データへのリンクを取得"""
        
        # arXivのancillary filesをチェック
        ancillary_url = f'https://arxiv.org/src/{arxiv_id}'
        
        # 論文ページから追加情報を取得
        paper_url = f'https://arxiv.org/abs/{arxiv_id}'
        
        try:
            response = requests.get(paper_url)
            if response.status_code == 200:
                # HTMLから追加のデータリンクを探す
                content = response.text
                
                # よくあるデータリポジトリのパターン
                patterns = {
                    'github': r'https?://github\.com/[\w-]+/[\w-]+',
                    'zenodo': r'https?://zenodo\.org/record/\d+',
                    'figshare': r'https?://figshare\.com/[\w/]+',
                    'dryad': r'https?://datadryad\.org/[\w/]+',
                    'osf': r'https?://osf\.io/[\w/]+'
                }
                
                found_links = {}
                for name, pattern in patterns.items():
                    matches = re.findall(pattern, content)
                    if matches:
                        found_links[name] = matches
                
                return found_links
        except:
            pass
        
        return {}
    
    def filter_papers_with_quantum_data(self, df):
        """量子実験データを含む可能性が高い論文をフィルタリング"""
        
        # スコアリングシステム
        def score_paper(row):
            score = 0
            
            # タイトルのキーワード
            title_lower = row['title'].lower()
            if 'experimental' in title_lower:
                score += 3
            if 'measurement' in title_lower:
                score += 2
            if 'benchmark' in title_lower:
                score += 2
            if 'characterization' in title_lower:
                score += 2
            
            # データ関連キーワードの数
            score += len(row['data_keywords']) * 2
            
            # GitHubリンクの存在
            if 'github_links' in row and row['github_links']:
                score += 5
            
            return score
        
        df['data_score'] = df.apply(score_paper, axis=1)
        
        # スコアが高い順にソート
        df_sorted = df.sort_values('data_score', ascending=False)
        
        # 上位の論文を返す
        return df_sorted[df_sorted['data_score'] > 0]

# 実行関数
def fetch_arxiv_quantum_experiments():
    """arXivから量子実験データを取得するメイン関数"""
    
    print("=== arXiv量子実験データ検索開始 ===")
    
    fetcher = ArxivQuantumDataFetcher()
    
    # 1. 論文を検索
    print("\n1. 関連論文を検索中...")
    papers_df = fetcher.search_quantum_datasets(max_results=200)
    print(f"   → 合計 {len(papers_df)} 件の論文を発見")
    
    # 2. データを含む可能性が高い論文をフィルタリング
    print("\n2. 実験データを含む論文をフィルタリング...")
    filtered_df = fetcher.filter_papers_with_quantum_data(papers_df)
    print(f"   → {len(filtered_df)} 件の有望な論文")
    
    # 3. 上位10件の詳細を表示
    print("\n3. 最も有望な論文（上位10件）:")
    print("-" * 80)
    
    for idx, row in filtered_df.head(10).iterrows():
        print(f"\n【{idx+1}】{row['title']}")
        print(f"   arXiv ID: {row['arxiv_id']}")
        print(f"   公開日: {row['published'][:10]}")
        print(f"   データスコア: {row['data_score']}")
        print(f"   データキーワード: {', '.join(row['data_keywords'])}")
        
        if 'github_links' in row and row['github_links']:
            print(f"   GitHub: {row['github_links']}")
        
        # 追加のデータリンクを取得
        extra_links = fetcher.get_supplementary_data_links(row['arxiv_id'])
        if extra_links:
            print(f"   追加リンク: {extra_links}")
    
    # 4. 結果を保存
    filtered_df.to_csv('arxiv_quantum_data/promising_papers.csv', index=False)
    
    # 5. 実際にデータをダウンロードできる論文のリストを作成
    downloadable_papers = create_downloadable_list(filtered_df)
    
    return filtered_df, downloadable_papers

def create_downloadable_list(df):
    """実際にデータをダウンロードできる論文のリストを作成"""
    
    downloadable = []
    
    for _, row in df.iterrows():
        if 'github_links' in row and row['github_links']:
            for github_link in row['github_links']:
                downloadable.append({
                    'title': row['title'],
                    'arxiv_id': row['arxiv_id'],
                    'data_source': 'github',
                    'url': f"https://{github_link}",
                    'type': 'repository'
                })
    
    # ダウンロード可能なデータソースを保存
    download_df = pd.DataFrame(downloadable)
    download_df.to_csv('arxiv_quantum_data/downloadable_sources.csv', index=False)
    
    print(f"\n4. ダウンロード可能なデータソース: {len(downloadable)} 件")
    
    return download_df

if __name__ == "__main__":
    # 実行
    papers, downloadable = fetch_arxiv_quantum_experiments()
    
    print("\n=== 次のステップ ===")
    print("1. promising_papers.csv を確認")
    print("2. downloadable_sources.csv からデータをダウンロード")
    print("3. download_quantum_datasets.py を実行してデータ取得")