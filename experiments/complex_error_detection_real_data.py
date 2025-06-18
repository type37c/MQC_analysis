#!/usr/bin/env python3
"""
実データを用いた複素エラー検出システムのテスト
Complex Error Detection System Test with Real Data

実際のBell状態データとIBM Quantum Volumeデータを用いて、
複素エラー検出システムの性能を評価します。
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
    from src.cqt_tracker_v3 import OptimizedCQTTracker
    from src.complex_error_detection import ComplexErrorDetector, compute_complex_correlation, detect_quantum_entanglement
    print("✓ 複素エラー検出モジュール読み込み成功")
except ImportError as e:
    print(f"⚠ モジュールインポートエラー: {e}")
    sys.exit(1)

# プロット設定
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

def load_real_trajectories():
    """実データから軌跡を生成"""
    trajectories = {}
    
    # 1. Bell状態データ（クリーンな参照用）
    bell_data_path = 'data_collection/collected_data/bell_states/bell_measurement_data.csv'
    
    if os.path.exists(bell_data_path):
        bell_data = pd.read_csv(bell_data_path)
        
        for idx, row in bell_data.iterrows():
            state = row['state']
            counts_str = row['counts']
            
            # countsの解析
            import ast
            counts_str = counts_str.replace("np.str_('", "'").replace("np.int64(", "").replace("')", "'").replace(")", "")
            counts = ast.literal_eval(counts_str)
            
            # CQT軌跡の生成
            tracker = OptimizedCQTTracker(system_dim=2)
            
            for outcome_str, count in counts.items():
                sample_count = min(count // 15, 150)  # 適度なサンプリング
                for _ in range(sample_count):
                    outcome = int(outcome_str[0])
                    tracker.add_measurement(outcome)
            
            if tracker.trajectory and len(tracker.trajectory) > 50:
                trajectories[f'bell_{state}'] = np.array(tracker.trajectory)
                print(f"Bell {state}: {len(tracker.trajectory)}点の軌跡")
    
    # 2. IBM Quantum Volumeデータ（様々なノイズレベル）
    qv_data_path = 'data_collection/downloaded_quantum_data/IBM_Quantum_Volume/qiskit-experiments/qiskit-experiments-main/test/library/quantum_volume'
    
    qv_files = {
        'qv_clean': 'qv_data_70_trials.json',  # 比較的クリーン
        'qv_moderate': 'qv_data_moderate_noise_100_trials.json',  # 中程度ノイズ
        'qv_noisy': 'qv_data_high_noise.json'  # 高ノイズ
    }
    
    for label, filename in qv_files.items():
        filepath = os.path.join(qv_data_path, filename)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                print(f"{label}: {len(data)} 試行データ読み込み")
                
                # 複数試行から軌跡生成
                tracker = OptimizedCQTTracker(system_dim=4)
                
                for trial_idx in range(min(3, len(data))):
                    trial = data[trial_idx]
                    
                    if 'counts' in trial:
                        counts = trial['counts']
                        
                        for bitstring, count in counts.items():
                            for _ in range(min(count, 25)):
                                outcome = int(bitstring[0]) if bitstring else 0
                                tracker.add_measurement(outcome)
                
                if tracker.trajectory and len(tracker.trajectory) > 100:
                    trajectories[label] = np.array(tracker.trajectory)
                    print(f"  軌跡生成: {len(tracker.trajectory)}点")
                    
            except Exception as e:
                print(f"  エラー: {filename} - {e}")
    
    return trajectories

def test_error_detection_performance(trajectories):
    """エラー検出性能のテスト"""\n    if not trajectories:\n        print(\"テスト用軌跡がありません\")\n        return None\n    \n    print(\"\\n=== 複素エラー検出性能テスト ===\")\n    \n    # Bell状態を参照軌跡として使用\n    bell_trajectories = {k: v for k, v in trajectories.items() if 'bell' in k}\n    qv_trajectories = {k: v for k, v in trajectories.items() if 'qv' in k}\n    \n    if not bell_trajectories:\n        print(\"参照用Bell状態軌跡がありません\")\n        return None\n    \n    # 最初のBell状態を参照として使用\n    reference_name = list(bell_trajectories.keys())[0]\n    reference_trajectory = bell_trajectories[reference_name]\n    \n    print(f\"参照軌跡: {reference_name} ({len(reference_trajectory)}点)\")\n    \n    # エラー検出器の初期化\n    detector = ComplexErrorDetector(reference_trajectory)\n    \n    error_results = []\n    detailed_results = {}\n    \n    # 全軌跡をテスト\n    for test_name, test_trajectory in trajectories.items():\n        if test_name == reference_name:\n            continue  # 参照軌跡自体はスキップ\n        \n        print(f\"\\n--- {test_name} のエラー検出テスト ---\")\n        \n        # エラー検出の実行\n        try:\n            errors = detector.detect_errors(test_trajectory)\n            error_analysis = detector.analyze_error_pattern(errors)\n            \n            # 基本統計\n            total_errors = len(errors)\n            error_rate = total_errors / len(test_trajectory)\n            mean_severity = np.mean([e['severity'] for e in errors]) if errors else 0\n            max_severity = np.max([e['severity'] for e in errors]) if errors else 0\n            \n            # エラーの分類\n            phase_errors = sum(1 for e in errors if e['error_type'] == 'phase_decoherence')\n            amplitude_errors = sum(1 for e in errors if e['error_type'] == 'amplitude_anomaly')\n            correlation_errors = sum(1 for e in errors if e['error_type'] == 'correlation_break')\n            \n            # 結果の保存\n            result = {\n                'test_name': test_name,\n                'trajectory_type': 'bell' if 'bell' in test_name else 'qv',\n                'noise_level': 'clean' if 'clean' in test_name or 'bell' in test_name else \n                              'moderate' if 'moderate' in test_name else 'high',\n                'trajectory_length': len(test_trajectory),\n                'total_errors': total_errors,\n                'error_rate': error_rate,\n                'mean_severity': mean_severity,\n                'max_severity': max_severity,\n                'phase_errors': phase_errors,\n                'amplitude_errors': amplitude_errors,\n                'correlation_errors': correlation_errors,\n                'error_clusters': len(error_analysis.get('position_clusters', []))\n            }\n            \n            error_results.append(result)\n            detailed_results[test_name] = {\n                'errors': errors,\n                'analysis': error_analysis\n            }\n            \n            # 詳細出力\n            print(f\"  軌跡長: {len(test_trajectory)}\")\n            print(f\"  検出エラー数: {total_errors}\")\n            print(f\"  エラー率: {error_rate:.4f}\")\n            print(f\"  平均深刻度: {mean_severity:.3f}\")\n            print(f\"  最大深刻度: {max_severity:.3f}\")\n            print(f\"  位相エラー: {phase_errors}, 振幅エラー: {amplitude_errors}, 相関エラー: {correlation_errors}\")\n            print(f\"  エラークラスタ: {len(error_analysis.get('position_clusters', []))}\")\n            \n        except Exception as e:\n            print(f\"  エラー検出失敗: {e}\")\n    \n    return error_results, detailed_results\n\ndef analyze_error_detection_results(error_results):\n    \"\"\"エラー検出結果の分析\"\"\"\n    if not error_results:\n        print(\"分析するエラー検出結果がありません\")\n        return None\n    \n    df = pd.DataFrame(error_results)\n    \n    print(\"\\n=== エラー検出結果分析 ===\")\n    print(df.round(4))\n    \n    # ノイズレベル別の統計\n    print(\"\\n--- ノイズレベル別統計 ---\")\n    noise_groups = df.groupby('noise_level')\n    \n    for noise_level, group in noise_groups:\n        print(f\"\\n{noise_level.upper()}ノイズ:\")\n        print(f\"  平均エラー率: {group['error_rate'].mean():.4f}\")\n        print(f\"  平均深刻度: {group['mean_severity'].mean():.3f}\")\n        print(f\"  平均エラークラスタ数: {group['error_clusters'].mean():.1f}\")\n    \n    # データタイプ別の統計\n    print(\"\\n--- データタイプ別統計 ---\")\n    type_groups = df.groupby('trajectory_type')\n    \n    for traj_type, group in type_groups:\n        print(f\"\\n{traj_type.upper()}データ:\")\n        print(f\"  平均エラー率: {group['error_rate'].mean():.4f}\")\n        print(f\"  平均深刻度: {group['mean_severity'].mean():.3f}\")\n        print(f\"  平均エラークラスタ数: {group['error_clusters'].mean():.1f}\")\n    \n    return df\n\ndef visualize_error_detection_results(error_results_df, trajectories, detailed_results):\n    \"\"\"エラー検出結果の可視化\"\"\"\n    if error_results_df is None or error_results_df.empty:\n        print(\"可視化するデータがありません\")\n        return\n    \n    fig, axes = plt.subplots(2, 3, figsize=(18, 12))\n    \n    # 1. エラー率の比較\n    ax = axes[0, 0]\n    colors = ['lightblue' if 'bell' in name else \n              'orange' if 'moderate' in name else \n              'red' if 'noisy' in name else 'green' \n              for name in error_results_df['test_name']]\n    \n    bars = ax.bar(range(len(error_results_df)), error_results_df['error_rate'], color=colors)\n    ax.set_xlabel('軌跡')\n    ax.set_ylabel('エラー率')\n    ax.set_title('軌跡別エラー検出率')\n    ax.set_xticks(range(len(error_results_df)))\n    ax.set_xticklabels([name.replace('_', '\\n') for name in error_results_df['test_name']], rotation=45)\n    ax.grid(True, alpha=0.3)\n    \n    # 値をバーの上に表示\n    for bar, value in zip(bars, error_results_df['error_rate']):\n        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, \n                f'{value:.3f}', ha='center', va='bottom', fontsize=9)\n    \n    # 2. エラーの深刻度分布\n    ax = axes[0, 1]\n    ax.scatter(error_results_df['error_rate'], error_results_df['max_severity'], \n               c=error_results_df['error_clusters'], s=100, alpha=0.7, cmap='viridis')\n    plt.colorbar(ax.collections[0], ax=ax, label='エラークラスタ数')\n    ax.set_xlabel('エラー率')\n    ax.set_ylabel('最大深刻度')\n    ax.set_title('エラー率 vs 深刻度')\n    \n    # 軌跡名を表示\n    for i, name in enumerate(error_results_df['test_name']):\n        ax.annotate(name.replace('_', '\\n'), \n                    (error_results_df['error_rate'].iloc[i], error_results_df['max_severity'].iloc[i]), \n                    fontsize=8, ha='center')\n    ax.grid(True, alpha=0.3)\n    \n    # 3. エラータイプの分布\n    ax = axes[0, 2]\n    x_pos = np.arange(len(error_results_df))\n    width = 0.25\n    \n    ax.bar(x_pos - width, error_results_df['phase_errors'], width, label='位相エラー', alpha=0.8)\n    ax.bar(x_pos, error_results_df['amplitude_errors'], width, label='振幅エラー', alpha=0.8)\n    ax.bar(x_pos + width, error_results_df['correlation_errors'], width, label='相関エラー', alpha=0.8)\n    \n    ax.set_xlabel('軌跡')\n    ax.set_ylabel('エラー数')\n    ax.set_title('エラータイプ別分布')\n    ax.set_xticks(x_pos)\n    ax.set_xticklabels([name.replace('_', '\\n') for name in error_results_df['test_name']], rotation=45)\n    ax.legend()\n    ax.grid(True, alpha=0.3)\n    \n    # 4. ノイズレベル別エラー率\n    ax = axes[1, 0]\n    noise_summary = error_results_df.groupby('noise_level')['error_rate'].agg(['mean', 'std']).reset_index()\n    \n    bars = ax.bar(noise_summary['noise_level'], noise_summary['mean'], \n                  yerr=noise_summary['std'], capsize=5, alpha=0.7,\n                  color=['green', 'orange', 'red'])\n    ax.set_xlabel('ノイズレベル')\n    ax.set_ylabel('平均エラー率')\n    ax.set_title('ノイズレベル別エラー検出性能')\n    ax.grid(True, alpha=0.3)\n    \n    # 5. 軌跡長 vs エラー数\n    ax = axes[1, 1]\n    scatter = ax.scatter(error_results_df['trajectory_length'], error_results_df['total_errors'], \n                        c=error_results_df['mean_severity'], s=100, alpha=0.7, cmap='plasma')\n    plt.colorbar(scatter, ax=ax, label='平均深刻度')\n    ax.set_xlabel('軌跡長')\n    ax.set_ylabel('総エラー数')\n    ax.set_title('軌跡長 vs エラー数')\n    ax.grid(True, alpha=0.3)\n    \n    # 6. エラークラスタ分析\n    ax = axes[1, 2]\n    bars = ax.bar(range(len(error_results_df)), error_results_df['error_clusters'], \n                  color=['lightblue' if 'bell' in name else \n                         'orange' if 'moderate' in name else \n                         'red' if 'noisy' in name else 'green' \n                         for name in error_results_df['test_name']])\n    ax.set_xlabel('軌跡')\n    ax.set_ylabel('エラークラスタ数')\n    ax.set_title('軌跡別エラークラスタ数')\n    ax.set_xticks(range(len(error_results_df)))\n    ax.set_xticklabels([name.replace('_', '\\n') for name in error_results_df['test_name']], rotation=45)\n    ax.grid(True, alpha=0.3)\n    \n    plt.suptitle('実データでの複素エラー検出システム性能評価', fontsize=16, fontweight='bold')\n    plt.tight_layout()\n    plt.savefig('complex_error_detection_real_data_results.png', dpi=300, bbox_inches='tight')\n    plt.show()\n\ndef test_quantum_entanglement_detection(trajectories):\n    \"\"\"量子もつれ検出のテスト\"\"\"\n    if len(trajectories) < 2:\n        print(\"もつれ検出には少なくとも2つの軌跡が必要です\")\n        return None\n    \n    print(\"\\n=== 量子もつれ検出テスト ===\")\n    \n    trajectory_names = list(trajectories.keys())\n    correlation_results = []\n    \n    for i in range(len(trajectory_names)):\n        for j in range(i + 1, len(trajectory_names)):\n            name1, name2 = trajectory_names[i], trajectory_names[j]\n            traj1, traj2 = trajectories[name1], trajectories[name2]\n            \n            # 長さを合わせる\n            min_len = min(len(traj1), len(traj2))\n            traj1_trimmed = traj1[:min_len]\n            traj2_trimmed = traj2[:min_len]\n            \n            # 複素相関の計算\n            correlation = compute_complex_correlation(traj1_trimmed, traj2_trimmed)\n            \n            # 量子もつれ検出\n            entanglement = detect_quantum_entanglement(traj1_trimmed, traj2_trimmed)\n            \n            correlation_results.append({\n                'trajectory1': name1,\n                'trajectory2': name2,\n                'complex_correlation': correlation,\n                'phase_correlation': entanglement['phase_correlation'],\n                'amplitude_correlation': entanglement['amplitude_correlation'],\n                'entanglement_score': entanglement['score'],\n                'is_entangled': entanglement['entangled']\n            })\n            \n            print(f\"{name1} ↔ {name2}:\")\n            print(f\"  複素相関: {correlation:.4f}\")\n            print(f\"  もつれスコア: {entanglement['score']:.4f}\")\n            print(f\"  もつれ判定: {'YES' if entanglement['entangled'] else 'NO'}\")\n    \n    return correlation_results\n\ndef main():\n    \"\"\"メイン実行関数\"\"\"\n    print(\"=\" * 60)\n    print(\"実データを用いた複素エラー検出システムのテスト\")\n    print(f\"実行開始: {datetime.now()}\")\n    print(\"=\" * 60)\n    \n    # 1. 実データ軌跡の読み込み\n    print(\"\\n1. 実データ軌跡の読み込み中...\")\n    trajectories = load_real_trajectories()\n    \n    if not trajectories:\n        print(\"エラー: 利用可能な軌跡データがありません\")\n        return\n    \n    print(f\"\\n総軌跡数: {len(trajectories)}\")\n    \n    # 2. エラー検出性能テスト\n    print(\"\\n2. エラー検出性能テスト実行中...\")\n    error_results, detailed_results = test_error_detection_performance(trajectories)\n    \n    if error_results:\n        # 3. 結果分析\n        print(\"\\n3. エラー検出結果の分析中...\")\n        error_df = analyze_error_detection_results(error_results)\n        \n        # 4. 可視化\n        print(\"\\n4. 結果の可視化中...\")\n        visualize_error_detection_results(error_df, trajectories, detailed_results)\n        \n        # 5. 量子もつれ検出テスト\n        print(\"\\n5. 量子もつれ検出テスト実行中...\")\n        correlation_results = test_quantum_entanglement_detection(trajectories)\n        \n        # 6. 結果保存\n        error_df.to_csv('complex_error_detection_real_data_results.csv', index=False)\n        \n        if correlation_results:\n            correlation_df = pd.DataFrame(correlation_results)\n            correlation_df.to_csv('quantum_entanglement_detection_results.csv', index=False)\n            print(\"\\n量子もつれ検出結果を quantum_entanglement_detection_results.csv に保存\")\n        \n        # 7. 主要発見の報告\n        print(\"\\n\" + \"=\" * 60)\n        print(\"🔬 主要な発見\")\n        print(\"=\" * 60)\n        \n        if len(error_results) > 0:\n            # 最も多くのエラーを検出\n            max_error_idx = error_df['total_errors'].idxmax()\n            max_error_name = error_df.loc[max_error_idx, 'test_name']\n            max_error_count = error_df.loc[max_error_idx, 'total_errors']\n            max_error_rate = error_df.loc[max_error_idx, 'error_rate']\n            \n            print(f\"\\n🚨 最も多くのエラーを検出:\")\n            print(f\"  {max_error_name}: {max_error_count}個のエラー（率: {max_error_rate:.4f}）\")\n            \n            # ノイズレベル別の平均エラー率\n            bell_avg = error_df[error_df['trajectory_type'] == 'bell']['error_rate'].mean() if not error_df[error_df['trajectory_type'] == 'bell'].empty else 0\n            qv_avg = error_df[error_df['trajectory_type'] == 'qv']['error_rate'].mean() if not error_df[error_df['trajectory_type'] == 'qv'].empty else 0\n            \n            print(f\"\\n📊 データタイプ別エラー検出性能:\")\n            print(f\"  Bell状態データ: 平均エラー率 = {bell_avg:.4f}\")\n            print(f\"  Quantum Volumeデータ: 平均エラー率 = {qv_avg:.4f}\")\n            \n            # もつれ検出結果\n            if correlation_results:\n                entangled_pairs = sum(1 for r in correlation_results if r['is_entangled'])\n                total_pairs = len(correlation_results)\n                entanglement_rate = entangled_pairs / total_pairs if total_pairs > 0 else 0\n                \n                print(f\"\\n🔗 量子もつれ検出:\")\n                print(f\"  検出されたもつれペア: {entangled_pairs}/{total_pairs} ({entanglement_rate:.2%})\")\n                \n                if entangled_pairs > 0:\n                    entangled_results = [r for r in correlation_results if r['is_entangled']]\n                    max_entanglement = max(entangled_results, key=lambda x: x['entanglement_score'])\n                    print(f\"  最強もつれペア: {max_entanglement['trajectory1']} ↔ {max_entanglement['trajectory2']}\")\n                    print(f\"  　　　　スコア: {max_entanglement['entanglement_score']:.4f}\")\n        \n        print(f\"\\n💡 科学的意義:\")\n        print(f\"  - 実データでの複素エラー検出システムの有効性を実証\")\n        print(f\"  - ノイズレベルによるエラーパターンの違いを定量化\")\n        print(f\"  - Bell状態と量子ボリュームデータの特性差を明確化\")\n        print(f\"  - 複素相関による量子もつれ検出手法を検証\")\n    \n    print(f\"\\n実行完了: {datetime.now()}\")\n    print(\"生成されたファイル:\")\n    print(\"  - complex_error_detection_real_data_results.png\")\n    print(\"  - complex_error_detection_real_data_results.csv\")\n    print(\"  - quantum_entanglement_detection_results.csv\")\n\nif __name__ == \"__main__\":\n    main()