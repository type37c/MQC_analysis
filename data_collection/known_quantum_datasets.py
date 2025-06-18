"""
既知の量子実験データセットのリスト
"""

KNOWN_QUANTUM_DATASETS = {
    "Google Quantum Supremacy": {
        "arxiv_id": "1910.11333",
        "title": "Quantum supremacy using a programmable superconducting processor",
        "data_url": "https://datadryad.org/stash/dataset/doi:10.5061/dryad.k6t1rj8",
        "description": "53量子ビットSycamoreプロセッサの実験データ"
    },
    
    "IBM Quantum Volume": {
        "arxiv_id": "2008.08571",
        "title": "Demonstration of quantum volume 64 on a superconducting quantum computing system",
        "github": "https://github.com/Qiskit/qiskit-experiments",
        "description": "量子ボリューム測定の実験データ"
    },
    
    "Bell State Tomography": {
        "arxiv_id": "1801.07904",
        "title": "Experimentally exploring compressed sensing quantum tomography",
        "github": "https://github.com/quantumlib/quantum-datasets",
        "description": "Bell状態トモグラフィーの実験データ"
    },
    
    "NISQ Benchmarks": {
        "arxiv_id": "2003.01293",
        "title": "Quantum Algorithm Implementations for Beginners",
        "supplementary": "https://github.com/lanl/quantum_algorithms",
        "description": "NISQ デバイスでのアルゴリズム実装データ"
    },
    
    "Quantum Error Characterization": {
        "arxiv_id": "2106.12627",
        "title": "Characterizing quantum instruments: from non-demolition measurements to quantum error correction",
        "zenodo": "https://zenodo.org/record/5012538",
        "description": "量子エラー特性の実験データ"
    }
}

def get_direct_download_links():
    """直接ダウンロード可能なリンクを返す"""
    
    direct_links = []
    
    for name, info in KNOWN_QUANTUM_DATASETS.items():
        if 'data_url' in info:
            direct_links.append({
                'name': name,
                'url': info['data_url'],
                'type': 'direct'
            })
        elif 'github' in info:
            direct_links.append({
                'name': name,
                'url': info['github'],
                'type': 'github'
            })
        elif 'zenodo' in info:
            direct_links.append({
                'name': name,
                'url': info['zenodo'],
                'type': 'zenodo'
            })
    
    return direct_links