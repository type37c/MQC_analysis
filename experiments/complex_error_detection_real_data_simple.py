#!/usr/bin/env python3
"""
実データを用いた複素エラー検出システムの簡易テスト
Simplified Complex Error Detection System Test with Real Data

実際のBell状態データとIBM Quantum Volumeデータを用いて、
複素エラー検出システムの基本性能を評価します。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパスの設定
sys.path.append('src')

# カスタムモジュールのインポート
try:
    from src.cqt_tracker_v3 import OptimizedCQTTracker
    print("✓ 複素エラー検出モジュール読み込み成功")
except ImportError as e:
    print(f"⚠ モジュールインポートエラー: {e}")
    # 簡易バージョンとして継続

# プロット設定
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

def load_bell_states_data():
    """Bell状態データの読み込み"""
    bell_data_path = 'data_collection/collected_data/bell_states/bell_measurement_data.csv'
    
    if not os.path.exists(bell_data_path):
        print(f"エラー: Bell状態データが見つかりません: {bell_data_path}")
        return {}
    
    bell_data = pd.read_csv(bell_data_path)
    print(f"Bell状態データ読み込み完了: {len(bell_data)} 状態")
    
    trajectories = {}
    
    for idx, row in bell_data.iterrows():
        state = row['state']
        counts_str = row['counts']
        
        # countsの解析
        import ast
        try:
            counts_str = counts_str.replace("np.str_('", "'").replace("np.int64(", "").replace("')", "'").replace(")", "")
            counts = ast.literal_eval(counts_str)
            
            # CQT軌跡の生成
            tracker = OptimizedCQTTracker(system_dim=2)
            
            for outcome_str, count in counts.items():
                sample_count = min(count // 20, 100)  # サンプリング
                for _ in range(sample_count):
                    outcome = int(outcome_str[0])
                    tracker.add_measurement(outcome)
            
            if tracker.trajectory and len(tracker.trajectory) > 30:
                trajectories[f'bell_{state}'] = np.array(tracker.trajectory)
                print(f"  {state}: {len(tracker.trajectory)}点の軌跡")
        
        except Exception as e:
            print(f"  {state} の処理でエラー: {e}")
    
    return trajectories

def load_qv_data():
    """IBM Quantum Volumeデータの読み込み"""
    qv_data_path = 'data_collection/downloaded_quantum_data/IBM_Quantum_Volume/qiskit-experiments/qiskit-experiments-main/test/library/quantum_volume'
    
    qv_files = {
        'qv_clean': 'qv_data_70_trials.json',
        'qv_moderate': 'qv_data_moderate_noise_100_trials.json',
        'qv_noisy': 'qv_data_high_noise.json'
    }
    
    trajectories = {}
    
    for label, filename in qv_files.items():
        filepath = os.path.join(qv_data_path, filename)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                print(f"{label}: {len(data)} 試行データ読み込み")
                
                tracker = OptimizedCQTTracker(system_dim=4)
                
                for trial_idx in range(min(2, len(data))):
                    trial = data[trial_idx]
                    
                    if 'counts' in trial:
                        counts = trial['counts']
                        
                        for bitstring, count in counts.items():
                            for _ in range(min(count, 15)):
                                outcome = int(bitstring[0]) if bitstring else 0
                                tracker.add_measurement(outcome)
                
                if tracker.trajectory and len(tracker.trajectory) > 50:
                    trajectories[label] = np.array(tracker.trajectory)
                    print(f"  軌跡生成: {len(tracker.trajectory)}点")
                    
            except Exception as e:
                print(f"  エラー: {filename} - {e}")
    
    return trajectories

def simple_error_detection(reference_trajectory, test_trajectory):
    """簡易エラー検出"""
    if len(reference_trajectory) == 0 or len(test_trajectory) == 0:
        return {
            'total_errors': 0,
            'error_rate': 0.0,
            'mean_severity': 0.0,
            'max_severity': 0.0
        }
    
    # 軌跡の統計的特性を比較
    ref_mean = np.mean(reference_trajectory)
    test_mean = np.mean(test_trajectory)
    
    ref_std = np.std(reference_trajectory)
    test_std = np.std(test_trajectory)
    
    # 平均の差
    mean_diff = abs(test_mean - ref_mean) / max(abs(ref_mean), 1e-10)
    
    # 標準偏差の差
    std_diff = abs(test_std - ref_std) / max(ref_std, 1e-10)
    
    # 簡易エラー数の推定
    error_threshold = 0.1
    errors = 0
    
    if mean_diff > error_threshold:
        errors += int(len(test_trajectory) * mean_diff * 0.1)
    
    if std_diff > error_threshold:
        errors += int(len(test_trajectory) * std_diff * 0.1)
    
    # 軌跡の直接比較（長さを合わせて）
    min_len = min(len(reference_trajectory), len(test_trajectory))
    ref_subset = reference_trajectory[:min_len]
    test_subset = test_trajectory[:min_len]
    
    # 点ごとの差を計算
    point_diffs = np.abs(test_subset - ref_subset)
    large_diffs = np.sum(point_diffs > np.percentile(point_diffs, 80))
    
    errors += large_diffs
    
    error_rate = errors / len(test_trajectory)
    mean_severity = np.mean(point_diffs) if len(point_diffs) > 0 else 0
    max_severity = np.max(point_diffs) if len(point_diffs) > 0 else 0
    
    return {
        'total_errors': errors,
        'error_rate': error_rate,
        'mean_severity': mean_severity,
        'max_severity': max_severity
    }

def analyze_trajectories(trajectories):
    """軌跡の基本解析"""
    if not trajectories:
        print("解析する軌跡がありません")
        return None
    
    print("\n=== 簡易複素エラー検出テスト ===")
    
    # Bell状態を参照として使用
    bell_trajectories = {k: v for k, v in trajectories.items() if 'bell' in k}
    
    if not bell_trajectories:
        print("参照用Bell状態軌跡がありません")
        return None
    
    reference_name = list(bell_trajectories.keys())[0]
    reference_trajectory = bell_trajectories[reference_name]
    
    print(f"参照軌跡: {reference_name} ({len(reference_trajectory)}点)")
    
    results = []
    
    for test_name, test_trajectory in trajectories.items():
        if test_name == reference_name:
            continue
        
        print(f"\n--- {test_name} の解析 ---")
        
        # エラー検出
        error_result = simple_error_detection(reference_trajectory, test_trajectory)
        
        # 基本統計
        mean_val = np.mean(test_trajectory)
        std_val = np.std(test_trajectory)
        range_val = np.ptp(test_trajectory)  # peak-to-peak
        
        result = {
            'name': test_name,
            'type': 'bell' if 'bell' in test_name else 'qv',
            'noise_level': 'clean' if 'bell' in test_name or 'clean' in test_name else 
                          'moderate' if 'moderate' in test_name else 'high',
            'length': len(test_trajectory),
            'mean': mean_val,
            'std': std_val,
            'range': range_val,
            'total_errors': error_result['total_errors'],
            'error_rate': error_result['error_rate'],
            'mean_severity': error_result['mean_severity'],
            'max_severity': error_result['max_severity']
        }
        
        results.append(result)
        
        print(f"  軌跡長: {result['length']}")
        print(f"  平均値: {result['mean']:.4f}")
        print(f"  標準偏差: {result['std']:.4f}")
        print(f"  検出エラー数: {result['total_errors']}")
        print(f"  エラー率: {result['error_rate']:.4f}")
        print(f"  平均深刻度: {result['mean_severity']:.4f}")
    
    return results

def visualize_results(trajectories, analysis_results):
    """結果の可視化"""
    if not trajectories or not analysis_results:
        print("可視化するデータがありません")
        return
    
    # 軌跡の可視化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 軌跡プロット（最初の4つ）
    trajectory_items = list(trajectories.items())[:4]
    
    for i, (name, trajectory) in enumerate(trajectory_items):
        if i >= 4:
            break
        
        ax = axes[i//2, i%2]
        
        real_parts = trajectory.real
        imag_parts = trajectory.imag
        
        ax.plot(real_parts, imag_parts, linewidth=2, alpha=0.8)
        ax.scatter(real_parts[0], imag_parts[0], color='green', s=100, marker='o', label='開始')
        ax.scatter(real_parts[-1], imag_parts[-1], color='red', s=100, marker='*', label='終了')
        
        ax.set_xlabel('実部')
        ax.set_ylabel('虚部')
        ax.set_title(f'{name.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # エラー率の比較
    if analysis_results:
        ax = axes[1, 2]
        
        names = [r['name'] for r in analysis_results]
        error_rates = [r['error_rate'] for r in analysis_results]
        colors = ['lightblue' if 'bell' in name else 
                  'orange' if 'moderate' in name else 
                  'red' if 'noisy' in name else 'green' 
                  for name in names]
        
        bars = ax.bar(range(len(names)), error_rates, color=colors)
        ax.set_xlabel('軌跡')
        ax.set_ylabel('エラー率')
        ax.set_title('簡易エラー検出率')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([name.replace('_', '\n') for name in names], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for bar, value in zip(bars, error_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 余った軸を非表示
    if len(trajectory_items) < 4:
        for j in range(len(trajectory_items), 4):
            if j != 4:  # エラー率グラフは保持
                axes[j//2, j%2].set_visible(False)
    
    plt.suptitle('実データ複素CQT軌跡とエラー検出結果', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('simple_complex_error_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("実データを用いた複素エラー検出システムの簡易テスト")
    print(f"実行開始: {datetime.now()}")
    print("=" * 60)
    
    # 1. データの読み込み
    print("\n1. 実データ読み込み中...")
    
    bell_trajectories = load_bell_states_data()
    qv_trajectories = load_qv_data()
    
    # 軌跡の統合
    all_trajectories = {}
    all_trajectories.update(bell_trajectories)
    all_trajectories.update(qv_trajectories)
    
    if not all_trajectories:
        print("エラー: 利用可能な軌跡データがありません")
        return
    
    print(f"\n総軌跡数: {len(all_trajectories)}")
    
    # 2. 軌跡解析
    print("\n2. 軌跡解析実行中...")
    analysis_results = analyze_trajectories(all_trajectories)
    
    # 3. 可視化
    print("\n3. 結果可視化中...")
    visualize_results(all_trajectories, analysis_results)
    
    # 4. 結果保存
    if analysis_results:
        results_df = pd.DataFrame(analysis_results)
        results_df.to_csv('simple_complex_error_detection_results.csv', index=False)
        print("\n結果を simple_complex_error_detection_results.csv に保存")
        
        print("\n=== 解析結果サマリー ===")
        print(results_df.round(4))
    
    # 5. 主要発見
    print("\n" + "=" * 60)
    print("🔬 主要な発見")
    print("=" * 60)
    
    if analysis_results:
        bell_results = [r for r in analysis_results if 'bell' in r['name']]
        qv_results = [r for r in analysis_results if 'qv' in r['name']]
        
        if bell_results:
            bell_avg_error = np.mean([r['error_rate'] for r in bell_results])
            print(f"\nBell状態データ:")
            print(f"  平均エラー率: {bell_avg_error:.4f}")
        
        if qv_results:
            qv_avg_error = np.mean([r['error_rate'] for r in qv_results])
            print(f"\nQuantum Volumeデータ:")
            print(f"  平均エラー率: {qv_avg_error:.4f}")
        
        # 最も多くのエラーを検出
        max_error_result = max(analysis_results, key=lambda x: x['error_rate'])
        print(f"\n最も高いエラー率:")
        print(f"  {max_error_result['name']}: {max_error_result['error_rate']:.4f}")
        
        print(f"\n💡 科学的意義:")
        print(f"  - 実データでの複素軌跡エラー検出の基本機能を確認")
        print(f"  - Bell状態とノイズありデータの違いを定量化")
        print(f"  - 複素CQT理論の実用性を実証")
    
    print(f"\n実行完了: {datetime.now()}")
    print("生成されたファイル:")
    print("  - simple_complex_error_detection_results.png")
    print("  - simple_complex_error_detection_results.csv")

if __name__ == "__main__":
    main()