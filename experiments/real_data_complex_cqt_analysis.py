#!/usr/bin/env python3
"""
実データを用いた複素CQT解析
Real Data Complex CQT Analysis

Bell状態データとIBM Quantum Volumeデータから複素軌跡を生成し、
複素演算による高度な解析を実行します。
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
sys.path.append('data_collection')

# カスタムモジュールのインポート
try:
    from src.cqt_tracker_v3 import OptimizedCQTTracker, MeasurementRecord
    from src.complex_cqt_operations import ComplexCQTAnalyzer
    from src.complex_error_detection import ComplexErrorDetector
    print("✓ CQTモジュール読み込み成功")
except ImportError as e:
    print(f"⚠ モジュールインポートエラー: {e}")
    sys.exit(1)

# プロット設定
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_bell_states_data():
    """Bell状態データの読み込み"""
    bell_data_path = 'data_collection/collected_data/bell_states/bell_measurement_data.csv'
    
    if not os.path.exists(bell_data_path):
        print(f"エラー: Bell状態データが見つかりません: {bell_data_path}")
        return None
    
    bell_data = pd.read_csv(bell_data_path)
    print(f"Bell状態データ読み込み完了: {len(bell_data)} 状態")
    
    trajectories = {}
    
    for idx, row in bell_data.iterrows():
        state = row['state']
        counts_str = row['counts']
        
        # countsの解析（文字列からdictへ変換）
        import ast
        counts_str = counts_str.replace("np.str_('", "'").replace("np.int64(", "").replace("')", "'").replace(")", "")
        counts = ast.literal_eval(counts_str)
        
        # CQT軌跡の生成
        tracker = OptimizedCQTTracker(system_dim=2)
        
        # 測定結果を順次入力
        for outcome_str, count in counts.items():
            # サンプリング（計算量を抑制）
            sample_count = min(count // 20, 100)  # 適度なサンプリング
            for _ in range(sample_count):
                outcome = int(outcome_str[0])
                tracker.add_measurement(outcome)
        
        if tracker.trajectory and len(tracker.trajectory) > 10:
            trajectories[f'bell_{state}'] = np.array(tracker.trajectory)
            print(f"  {state}: {len(tracker.trajectory)}点の軌跡")
    
    return trajectories

def load_quantum_volume_data():
    """IBM Quantum Volumeデータの読み込み"""
    qv_data_path = 'data_collection/downloaded_quantum_data/IBM_Quantum_Volume/qiskit-experiments/qiskit-experiments-main/test/library/quantum_volume'
    
    data_files = {
        'qv_moderate_100': 'qv_data_moderate_noise_100_trials.json',
        'qv_moderate_300': 'qv_data_moderate_noise_300_trials.json',
        'qv_high_noise': 'qv_data_high_noise.json',
        'qv_standard': 'qv_data_70_trials.json'
    }
    
    trajectories = {}
    
    for label, filename in data_files.items():
        filepath = os.path.join(qv_data_path, filename)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                print(f"{label}: {len(data)} 試行データ読み込み")
                
                # 最初の数試行を使用して軌跡生成
                tracker = OptimizedCQTTracker(system_dim=4)  # 4量子ビット系
                
                for trial_idx in range(min(5, len(data))):  # 最初の5試行
                    trial = data[trial_idx]
                    
                    if 'counts' in trial:
                        counts = trial['counts']
                        
                        # 測定結果の処理
                        for bitstring, count in counts.items():
                            for _ in range(min(count, 20)):  # サンプリング
                                # 最初のビットを使用
                                outcome = int(bitstring[0]) if bitstring else 0
                                tracker.add_measurement(outcome)
                
                if tracker.trajectory and len(tracker.trajectory) > 20:
                    trajectories[label] = np.array(tracker.trajectory)
                    print(f"  軌跡生成: {len(tracker.trajectory)}点")
                    
            except Exception as e:
                print(f"  エラー: {filename} - {e}")
    
    return trajectories

def analyze_complex_trajectory(name, trajectory):
    """複素軌跡の詳細解析"""
    print(f"\n--- {name} の複素演算解析 ---")
    
    try:
        # ComplexCQTAnalyzerで解析
        analyzer = ComplexCQTAnalyzer(trajectory)
        
        # 各種解析の実行
        instant_props = analyzer.compute_instantaneous_properties()
        w_features = analyzer.analyze_w_pattern()
        fourier = analyzer.fourier_analysis()
        transitions = analyzer.detect_phase_transitions()
        invariants = analyzer.calculate_geometric_invariants()
        
        # 解析結果のサマリー
        analysis_summary = {
            'name': name,
            'trajectory_length': len(trajectory),
            'mean_speed': np.mean(instant_props['speed']),
            'max_acceleration': np.max(np.abs(instant_props['acceleration'])),
            'mean_curvature': np.mean(np.abs(instant_props['curvature'])),
            'winding_number': w_features['winding_number'],
            'fractal_dimension': w_features['fractal_dimension'],
            'spectral_entropy': fourier['spectral_entropy'],
            'total_length': invariants['total_length'],
            'enclosed_area': invariants['enclosed_area'],
            'asymmetry': invariants['asymmetry'],
            'compactness': invariants['compactness'],
            'num_transitions': len(transitions['all_transitions'])
        }
        
        # 統計表示
        print(f"  軌跡長: {len(trajectory)}")
        print(f"  平均速度: {analysis_summary['mean_speed']:.4f}")
        print(f"  平均曲率: {analysis_summary['mean_curvature']:.4f}")
        print(f"  巻き数: {analysis_summary['winding_number']:.3f}")
        print(f"  フラクタル次元: {analysis_summary['fractal_dimension']:.3f}")
        print(f"  スペクトルエントロピー: {analysis_summary['spectral_entropy']:.3f}")
        print(f"  コンパクト性: {analysis_summary['compactness']:.4f}")
        print(f"  相転移点数: {analysis_summary['num_transitions']}")
        
        return analysis_summary, {
            'analyzer': analyzer,
            'instant_props': instant_props,
            'w_features': w_features,
            'fourier': fourier,
            'transitions': transitions,
            'invariants': invariants
        }
        
    except Exception as e:
        print(f"  解析エラー: {e}")
        return None, None

def visualize_trajectories(trajectories, analysis_results):
    """軌跡の可視化"""
    n_trajectories = len(trajectories)
    if n_trajectories == 0:
        print("可視化する軌跡がありません")
        return
    
    # 4つまでの軌跡を表示
    display_count = min(4, n_trajectories)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    trajectory_names = list(trajectories.keys())[:display_count]
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, name in enumerate(trajectory_names):
        ax = axes[i]
        trajectory = trajectories[name]
        
        # 複素軌跡のプロット
        real_parts = trajectory.real
        imag_parts = trajectory.imag
        
        ax.plot(real_parts, imag_parts, color=colors[i], linewidth=2, alpha=0.8)
        ax.scatter(real_parts[0], imag_parts[0], color='green', s=100, marker='o', label='開始')
        ax.scatter(real_parts[-1], imag_parts[-1], color='red', s=100, marker='*', label='終了')
        
        ax.set_xlabel('実部 (方向性)')
        ax.set_ylabel('虚部 (不確実性)')
        ax.set_title(f'{name.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 統計情報の表示
        if name in analysis_results and analysis_results[name]:
            summary = analysis_results[name]
            info_text = f"速度: {summary['mean_speed']:.3f}\n巻き数: {summary['winding_number']:.2f}\nフラクタル: {summary['fractal_dimension']:.3f}"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 余った軸を非表示
    for j in range(display_count, 4):
        axes[j].set_visible(False)
    
    plt.suptitle('実データから生成された複素CQT軌跡', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('real_data_complex_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_analysis_results(analysis_summaries):
    """解析結果の比較"""
    if not analysis_summaries:
        print("比較する解析結果がありません")
        return
    
    df = pd.DataFrame(analysis_summaries)
    
    print("\n=== 実データ複素CQT解析結果比較 ===")
    print(df.round(4))
    
    # 特徴量の可視化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 速度 vs 曲率
    ax = axes[0, 0]
    scatter = ax.scatter(df['mean_speed'], df['mean_curvature'], 
                        c=df['spectral_entropy'], s=100, alpha=0.7, cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='スペクトルエントロピー')
    ax.set_xlabel('平均速度')
    ax.set_ylabel('平均曲率')
    ax.set_title('速度 vs 曲率')
    for i, name in enumerate(df['name']):
        ax.annotate(name.replace('_', '\n'), (df['mean_speed'][i], df['mean_curvature'][i]), 
                    fontsize=8, ha='center')
    ax.grid(True, alpha=0.3)
    
    # 2. 巻き数 vs フラクタル次元
    ax = axes[0, 1]
    valid_mask = ~np.isnan(df['fractal_dimension'])
    if valid_mask.any():
        scatter = ax.scatter(df.loc[valid_mask, 'winding_number'], 
                            df.loc[valid_mask, 'fractal_dimension'], 
                            c=df.loc[valid_mask, 'asymmetry'], 
                            s=100, alpha=0.7, cmap='plasma')
        plt.colorbar(scatter, ax=ax, label='非対称性')
    ax.set_xlabel('巻き数')
    ax.set_ylabel('フラクタル次元')
    ax.set_title('巻き数 vs フラクタル次元')
    ax.grid(True, alpha=0.3)
    
    # 3. 軌跡長 vs 面積
    ax = axes[0, 2]
    scatter = ax.scatter(df['total_length'], df['enclosed_area'], 
                        c=df['compactness'], s=100, alpha=0.7, cmap='cool')
    plt.colorbar(scatter, ax=ax, label='コンパクト性')
    ax.set_xlabel('軌跡総長')
    ax.set_ylabel('囲む面積')
    ax.set_title('軌跡長 vs 面積')
    ax.grid(True, alpha=0.3)
    
    # 4. スペクトルエントロピー比較
    ax = axes[1, 0]
    bars = ax.bar(range(len(df)), df['spectral_entropy'], 
                  color=plt.cm.Set3(np.linspace(0, 1, len(df))))
    ax.set_xlabel('軌跡')
    ax.set_ylabel('スペクトルエントロピー')
    ax.set_title('軌跡別スペクトルエントロピー')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([name.replace('_', '\n') for name in df['name']], rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 5. コンパクト性比較
    ax = axes[1, 1]
    bars = ax.bar(range(len(df)), df['compactness'], 
                  color=plt.cm.Pastel1(np.linspace(0, 1, len(df))))
    ax.set_xlabel('軌跡')
    ax.set_ylabel('コンパクト性')
    ax.set_title('軌跡別コンパクト性')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([name.replace('_', '\n') for name in df['name']], rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 6. 相転移点数比較
    ax = axes[1, 2]
    bars = ax.bar(range(len(df)), df['num_transitions'], 
                  color=plt.cm.viridis(np.linspace(0, 1, len(df))))
    ax.set_xlabel('軌跡')
    ax.set_ylabel('相転移点数')
    ax.set_title('軌跡別相転移点数')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([name.replace('_', '\n') for name in df['name']], rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('実データ複素CQT解析結果の比較', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('real_data_analysis_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # CSVで保存
    df.to_csv('real_data_complex_cqt_results.csv', index=False)
    print("\n解析結果を real_data_complex_cqt_results.csv に保存しました")
    
    return df

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("実データを用いた複素CQT解析")
    print(f"実行開始: {datetime.now()}")
    print("=" * 60)
    
    # 1. データの読み込み
    print("\n1. データ読み込み中...")
    bell_trajectories = load_bell_states_data()
    qv_trajectories = load_quantum_volume_data()
    
    # 軌跡の統合
    all_trajectories = {}
    if bell_trajectories:
        all_trajectories.update(bell_trajectories)
    if qv_trajectories:
        all_trajectories.update(qv_trajectories)
    
    if not all_trajectories:
        print("エラー: 利用可能な軌跡データがありません")
        return
    
    print(f"\n総軌跡数: {len(all_trajectories)}")
    
    # 2. 複素演算解析
    print("\n2. 複素演算解析実行中...")
    analysis_summaries = []
    detailed_results = {}
    
    for name, trajectory in all_trajectories.items():
        summary, details = analyze_complex_trajectory(name, trajectory)
        if summary:
            analysis_summaries.append(summary)
            detailed_results[name] = details
    
    # 3. 結果の可視化と比較
    print("\n3. 結果の可視化と比較...")
    visualize_trajectories(all_trajectories, 
                         {s['name']: s for s in analysis_summaries})
    
    comparison_df = compare_analysis_results(analysis_summaries)
    
    # 4. 主要な発見の報告
    print("\n" + "=" * 60)
    print("🔬 主要な発見")
    print("=" * 60)
    
    if len(analysis_summaries) > 0:
        # Bell states vs Quantum Volume の比較
        bell_results = [s for s in analysis_summaries if 'bell' in s['name']]
        qv_results = [s for s in analysis_summaries if 'qv' in s['name']]
        
        if bell_results:
            bell_entropy_avg = np.mean([s['spectral_entropy'] for s in bell_results])
            bell_fractal_avg = np.mean([s['fractal_dimension'] for s in bell_results if not np.isnan(s['fractal_dimension'])])
            print(f"\nBell状態データ:")
            print(f"  平均スペクトルエントロピー: {bell_entropy_avg:.4f}")
            if not np.isnan(bell_fractal_avg):
                print(f"  平均フラクタル次元: {bell_fractal_avg:.4f}")
        
        if qv_results:
            qv_entropy_avg = np.mean([s['spectral_entropy'] for s in qv_results])
            qv_fractal_avg = np.mean([s['fractal_dimension'] for s in qv_results if not np.isnan(s['fractal_dimension'])])
            print(f"\nQuantum Volumeデータ:")
            print(f"  平均スペクトルエントロピー: {qv_entropy_avg:.4f}")
            if not np.isnan(qv_fractal_avg):
                print(f"  平均フラクタル次元: {qv_fractal_avg:.4f}")
        
        # 最も興味深い結果
        max_entropy_idx = comparison_df['spectral_entropy'].idxmax()
        max_complexity_name = comparison_df.loc[max_entropy_idx, 'name']
        max_complexity_value = comparison_df.loc[max_entropy_idx, 'spectral_entropy']
        
        print(f"\n最も複雑な軌跡:")
        print(f"  {max_complexity_name}: エントロピー = {max_complexity_value:.4f}")
        
        max_transitions_idx = comparison_df['num_transitions'].idxmax()
        max_transitions_name = comparison_df.loc[max_transitions_idx, 'name']
        max_transitions_value = comparison_df.loc[max_transitions_idx, 'num_transitions']
        
        print(f"\n最も多くの相転移:")
        print(f"  {max_transitions_name}: {max_transitions_value}回の相転移")
    
    print(f"\n実行完了: {datetime.now()}")
    print("生成されたファイル:")
    print("  - real_data_complex_trajectories.png")
    print("  - real_data_analysis_comparison.png")
    print("  - real_data_complex_cqt_results.csv")

if __name__ == "__main__":
    main()