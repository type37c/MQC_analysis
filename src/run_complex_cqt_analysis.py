"""
複素演算を活用したCQT解析の実行
"""
import numpy as np
import pandas as pd
import os
import sys
import json
from datetime import datetime

# プロジェクトパスの設定
sys.path.append('/home/type37c/projects/CQT_experiments/src')
sys.path.append('/home/type37c/projects/CQT_experiments/data_collection')

from complex_cqt_operations import ComplexCQTAnalyzer, run_complex_analysis, visualize_complex_analysis
from complex_error_detection import ComplexErrorDetector, compute_complex_correlation, detect_quantum_entanglement
from cqt_tracker_v3 import OptimizedCQTTracker

def load_existing_trajectory_data():
    """既存の軌跡データを読み込み"""
    trajectories = {}
    
    # 1. シミュレーションデータからの軌跡
    bell_data_path = '/home/type37c/projects/CQT_experiments/data_collection/collected_data/bell_states/bell_measurement_data.csv'
    
    if os.path.exists(bell_data_path):
        print("Bell状態データから複素軌跡を生成...")
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
            
            # 測定結果を順次入力
            for outcome_str, count in counts.items():
                # ビット列から測定結果を生成
                for _ in range(min(count // 10, 100)):  # サンプリング
                    outcome = int(outcome_str[0])
                    tracker.add_measurement(outcome)
            
            if tracker.trajectory:
                trajectories[f'bell_{state}'] = tracker.trajectory
    
    # 2. 実量子データからの軌跡
    qv_data_path = '/home/type37c/projects/CQT_experiments/data_collection/downloaded_quantum_data/IBM_Quantum_Volume/qiskit-experiments/qiskit-experiments-main/test/library/quantum_volume'
    
    qv_files = {
        'qv_moderate_100': 'qv_data_moderate_noise_100_trials.json',
        'qv_moderate_300': 'qv_data_moderate_noise_300_trials.json',
        'qv_high_noise': 'qv_data_high_noise.json',
        'qv_standard': 'qv_data_70_trials.json'
    }
    
    for label, filename in qv_files.items():
        filepath = os.path.join(qv_data_path, filename)
        
        if os.path.exists(filepath):
            print(f"{label}データから複素軌跡を生成...")
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # 最初の3試行から軌跡を生成
            for trial_idx in range(min(3, len(data))):
                trial = data[trial_idx]
                
                if 'counts' in trial:
                    tracker = OptimizedCQTTracker(system_dim=2)
                    counts = trial['counts']
                    
                    # 測定結果の処理
                    for bitstring, count in counts.items():
                        for _ in range(min(count, 20)):  # サンプリング
                            outcome = int(bitstring[0]) if bitstring else 0
                            tracker.add_measurement(outcome)
                    
                    if tracker.trajectory:
                        trajectories[f'{label}_trial_{trial_idx}'] = tracker.trajectory
    
    return trajectories

def generate_synthetic_w_pattern():
    """W字パターンの合成軌跡を生成"""
    print("W字パターンの合成軌跡を生成...")
    
    # W字の基本形状を複素数で表現
    t = np.linspace(0, 4*np.pi, 200)
    
    # W字の実部（3つの山）
    real_part = np.sin(t) * np.sin(3*t/4)
    
    # W字の虚部（不確実性の変化）
    imag_part = 0.3 * np.sin(2*t) + 0.2 * np.cos(t/2)
    
    # ノイズを追加
    noise_real = 0.1 * np.random.normal(0, 1, len(t))
    noise_imag = 0.05 * np.random.normal(0, 1, len(t))
    
    w_trajectory = (real_part + noise_real) + 1j * (imag_part + noise_imag)
    
    return w_trajectory

def create_error_test_trajectories():
    """エラー検出テスト用の軌跡を生成"""
    print("エラー検出テスト用軌跡を生成...")
    
    # 1. クリーンな軌跡
    clean_trajectory = generate_synthetic_w_pattern()
    
    # 2. ビットフリップエラーを含む軌跡
    bitflip_trajectory = clean_trajectory.copy()
    error_positions = np.random.choice(len(bitflip_trajectory), size=len(bitflip_trajectory)//20, replace=False)
    for pos in error_positions:
        # 実部の符号を反転（ビットフリップをシミュレート）
        bitflip_trajectory[pos] = -bitflip_trajectory[pos].real + 1j * bitflip_trajectory[pos].imag
    
    # 3. 位相ノイズを含む軌跡
    phase_noise_trajectory = clean_trajectory.copy()
    phase_noise = 0.5 * np.random.normal(0, 1, len(phase_noise_trajectory))
    for i, noise in enumerate(phase_noise):
        magnitude = abs(phase_noise_trajectory[i])
        phase = np.angle(phase_noise_trajectory[i]) + noise
        phase_noise_trajectory[i] = magnitude * np.exp(1j * phase)
    
    # 4. 振幅減衰を含む軌跡
    amplitude_decay_trajectory = clean_trajectory.copy()
    decay_factor = np.exp(-np.linspace(0, 2, len(amplitude_decay_trajectory)))
    amplitude_decay_trajectory *= decay_factor
    
    return {
        'clean': clean_trajectory,
        'bitflip': bitflip_trajectory,
        'phase_noise': phase_noise_trajectory,
        'amplitude_decay': amplitude_decay_trajectory
    }

def main():
    print("=== 複素演算によるCQT解析開始 ===")
    print(f"開始時刻: {datetime.now()}")
    
    # 1. 既存データの読み込み
    print("\n1. 既存データの読み込み...")
    trajectories = load_existing_trajectory_data()
    print(f"読み込まれた軌跡数: {len(trajectories)}")
    
    # 2. 合成W字パターンの生成
    print("\n2. 合成W字パターンの生成...")
    w_pattern = generate_synthetic_w_pattern()
    trajectories['synthetic_w_pattern'] = w_pattern
    
    # 3. 各軌跡の複素演算解析
    print("\n3. 複素演算解析の実行...")
    analysis_results = {}
    
    for name, trajectory in trajectories.items():
        if len(trajectory) > 10:  # 最小長チェック
            print(f"\n--- {name} の解析中 ---")
            try:
                analyzer = run_complex_analysis(trajectory, name=name)
                analysis_results[name] = {
                    'analyzer': analyzer,
                    'w_features': analyzer.analyze_w_pattern(),
                    'fourier': analyzer.fourier_analysis(),
                    'invariants': analyzer.calculate_geometric_invariants()
                }
            except Exception as e:
                print(f"エラー: {name} の解析に失敗: {e}")
    
    # 4. エラー検出の実行
    print("\n4. エラー検出の実行...")
    error_test_trajectories = create_error_test_trajectories()
    
    # クリーンな軌跡を参照として使用
    reference_trajectory = error_test_trajectories['clean']
    detector = ComplexErrorDetector(reference_trajectory)
    
    error_results = {}
    for test_name, test_trajectory in error_test_trajectories.items():
        if test_name != 'clean':  # クリーンな軌跡以外をテスト
            print(f"\n--- {test_name} のエラー検出中 ---")
            errors = detector.detect_errors(test_trajectory)
            error_results[test_name] = errors
            
            print(f"検出されたエラー数: {len(errors)}")
            
            # エラー検出の可視化
            detector.visualize_error_detection(test_trajectory, errors, 
                                             save_path=f'error_detection_{test_name}.png')
            
            # エラーレポートの生成
            detector.generate_error_report(test_trajectory, errors, 
                                         save_path=f'error_report_{test_name}.txt')
    
    # 5. 複素相関解析
    print("\n5. 複素相関解析...")
    if len(trajectories) > 1:
        trajectory_names = list(trajectories.keys())
        for i in range(len(trajectory_names)):
            for j in range(i+1, len(trajectory_names)):
                name1, name2 = trajectory_names[i], trajectory_names[j]
                traj1, traj2 = trajectories[name1], trajectories[name2]
                
                # 長さを合わせる
                min_len = min(len(traj1), len(traj2))
                traj1_trimmed = traj1[:min_len]
                traj2_trimmed = traj2[:min_len]
                
                correlation = compute_complex_correlation(traj1_trimmed, traj2_trimmed)
                print(f"{name1} vs {name2}: 複素相関 = {correlation:.4f}")
                
                # 量子もつれ検出
                entanglement = detect_quantum_entanglement(traj1_trimmed, traj2_trimmed)
                if entanglement['entangled']:
                    print(f"  → 量子もつれ検出! スコア: {entanglement['score']:.4f}")
    
    # 6. 比較分析とサマリー
    print("\n6. 比較分析とサマリー...")
    
    # 特徴量の比較テーブル
    feature_comparison = []
    for name, result in analysis_results.items():
        w_features = result['w_features']
        fourier = result['fourier']
        invariants = result['invariants']
        
        feature_comparison.append({
            'name': name,
            'winding_number': w_features['winding_number'],
            'fractal_dimension': w_features['fractal_dimension'],
            'spectral_entropy': fourier['spectral_entropy'],
            'total_length': invariants['total_length'],
            'enclosed_area': invariants['enclosed_area'],
            'asymmetry': invariants['asymmetry']
        })
    
    comparison_df = pd.DataFrame(feature_comparison)
    print("\n=== 特徴量比較テーブル ===")
    print(comparison_df.round(4))
    
    # 結果をCSVで保存
    comparison_df.to_csv('complex_cqt_feature_comparison.csv', index=False)
    print("\n特徴量比較テーブルをCSVで保存しました: complex_cqt_feature_comparison.csv")
    
    # 7. 最終サマリー
    print(f"\n=== 複素演算CQT解析完了 ===")
    print(f"終了時刻: {datetime.now()}")
    print(f"解析された軌跡数: {len(analysis_results)}")
    print(f"エラー検出テスト数: {len(error_results)}")
    print(f"生成された可視化ファイル数: {len(analysis_results) + len(error_results)}")
    
    # 主要な発見
    print("\n=== 主要な発見 ===")
    if analysis_results:
        # 最も複雑な軌跡
        max_entropy = max(analysis_results.items(), 
                         key=lambda x: x[1]['fourier']['spectral_entropy'])
        print(f"最も複雑な軌跡: {max_entropy[0]} (エントロピー: {max_entropy[1]['fourier']['spectral_entropy']:.3f})")
        
        # 最も規則的な軌跡
        min_entropy = min(analysis_results.items(), 
                         key=lambda x: x[1]['fourier']['spectral_entropy'])
        print(f"最も規則的な軌跡: {min_entropy[0]} (エントロピー: {min_entropy[1]['fourier']['spectral_entropy']:.3f})")
        
        # フラクタル次元の範囲
        fractal_dims = [result['w_features']['fractal_dimension'] 
                       for result in analysis_results.values() 
                       if result['w_features']['fractal_dimension'] is not None]
        if fractal_dims:
            print(f"フラクタル次元の範囲: {min(fractal_dims):.3f} - {max(fractal_dims):.3f}")
    
    print("\n🎯 複素演算によるCQT理論の深化解析が完了しました！")

if __name__ == "__main__":
    main()