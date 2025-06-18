"""
収集したデータを解析する最初のステップ
CQT理論による複素数解析への準備
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import os

def load_collected_data():
    """収集されたデータを読み込む"""
    data_files = {
        'bell_states': 'collected_data/bell_states/bell_measurement_data.csv',
        'rotation_sweep': 'collected_data/custom_experiments/rotation_sweep_data.csv',
        'noise_characterization': 'collected_data/error_characterization/noise_sweep_data.csv',
        'vqe_landscape': 'collected_data/vqe_results/vqe_landscape_H2.csv'
    }
    
    loaded_data = {}
    
    for name, filepath in data_files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            loaded_data[name] = df
            print(f"✓ {name}: {df.shape[0]} records loaded")
        else:
            print(f"✗ {name}: file not found at {filepath}")
    
    return loaded_data

def analyze_bell_data(bell_data):
    """Bell状態データの詳細解析"""
    print("\n=== Bell状態データの構造解析 ===")
    print(f"測定状態数: {len(bell_data)}")
    print(f"総ショット数: {bell_data['shots'].sum()}")
    print(f"状態の種類: {list(bell_data['state'].unique())}")
    
    # counts カラムの解析
    print("\n=== 測定結果の詳細 ===")
    
    for idx, row in bell_data.iterrows():
        state = row['state']
        counts_str = row['counts']
        shots = row['shots']
        
        # 文字列として保存されたdictを復元
        try:
            # NumPy型の文字列表現を処理
            counts_str = counts_str.replace("np.str_('", "'").replace("np.int64(", "").replace("')", "'").replace(")", "")
            counts = ast.literal_eval(counts_str)
            
            print(f"\n{state}:")
            print(f"  総測定数: {shots}")
            print(f"  測定結果: {counts}")
            
            # 確率計算
            total_counts = sum(counts.values())
            probabilities = {outcome: count/total_counts for outcome, count in counts.items()}
            print(f"  確率分布: {probabilities}")
            
            # Bell状態の理論値との比較
            if state in ['phi_plus', 'phi_minus']:
                # |Φ±⟩ = (|00⟩ ± |11⟩)/√2
                expected_prob = 0.5
                actual_prob_00 = probabilities.get('00', 0)
                actual_prob_11 = probabilities.get('11', 0)
                
                print(f"  理論値: |00⟩=0.5, |11⟩=0.5")
                print(f"  実測値: |00⟩={actual_prob_00:.3f}, |11⟩={actual_prob_11:.3f}")
                print(f"  誤差: |00⟩={abs(actual_prob_00-0.5):.4f}, |11⟩={abs(actual_prob_11-0.5):.4f}")
                
            elif state in ['psi_plus', 'psi_minus']:
                # |Ψ±⟩ = (|01⟩ ± |10⟩)/√2
                expected_prob = 0.5
                actual_prob_01 = probabilities.get('01', 0)
                actual_prob_10 = probabilities.get('10', 0)
                
                print(f"  理論値: |01⟩=0.5, |10⟩=0.5")
                print(f"  実測値: |01⟩={actual_prob_01:.3f}, |10⟩={actual_prob_10:.3f}")
                print(f"  誤差: |01⟩={abs(actual_prob_01-0.5):.4f}, |10⟩={abs(actual_prob_10-0.5):.4f}")
            
        except Exception as e:
            print(f"  データ解析エラー: {e}")
            print(f"  生の counts データ: {counts_str}")

def analyze_rotation_data(rotation_data):
    """回転スイープデータの解析"""
    print("\n=== 回転スイープデータの解析 ===")
    print(f"データポイント数: {len(rotation_data)}")
    print(f"角度範囲: {rotation_data['angle'].min():.3f} - {rotation_data['angle'].max():.3f} rad")
    
    # 理論値と実測値の比較
    mean_error = np.mean(np.abs(rotation_data['probability_1'] - rotation_data['theoretical_prob_1']))
    max_error = np.max(np.abs(rotation_data['probability_1'] - rotation_data['theoretical_prob_1']))
    
    print(f"平均測定誤差: {mean_error:.4f}")
    print(f"最大測定誤差: {max_error:.4f}")
    
    # 統計的品質評価
    correlation = np.corrcoef(rotation_data['probability_1'], rotation_data['theoretical_prob_1'])[0,1]
    print(f"理論値との相関係数: {correlation:.4f}")
    
    return {
        'mean_error': mean_error,
        'max_error': max_error,
        'correlation': correlation
    }

def analyze_noise_data(noise_data):
    """ノイズ特性データの解析"""
    print("\n=== ノイズ特性データの解析 ===")
    print(f"ノイズレベル数: {len(noise_data)}")
    print(f"ノイズ範囲: {noise_data['noise_level'].min():.3f} - {noise_data['noise_level'].max():.3f}")
    
    # 忠実度の劣化分析
    for idx, row in noise_data.iterrows():
        noise_level = row['noise_level']
        fidelity = row['fidelity']
        print(f"  ノイズ {noise_level:.3f}: 忠実度 {fidelity:.4f}")
    
    # ノイズ耐性の閾値分析
    high_fidelity_threshold = 0.95
    acceptable_noise = noise_data[noise_data['fidelity'] >= high_fidelity_threshold]
    
    if not acceptable_noise.empty:
        max_acceptable_noise = acceptable_noise['noise_level'].max()
        print(f"\n高忠実度維持可能な最大ノイズレベル: {max_acceptable_noise:.3f}")
    else:
        print("\n高忠実度を維持するノイズレベルが見つかりません")

def prepare_for_cqt_analysis(loaded_data):
    """CQT解析に向けたデータ準備状況の評価"""
    print("\n=== CQT解析への準備状況 ===")
    
    quality_score = 0
    max_score = 4
    
    # 1. Bell状態データの品質
    if 'bell_states' in loaded_data:
        bell_data = loaded_data['bell_states']
        if len(bell_data) == 4:  # 4つのBell状態全て
            quality_score += 1
            print("✓ Bell状態データ: 4状態完備")
        else:
            print("✗ Bell状態データ: 不完全")
    else:
        print("✗ Bell状態データ: 未収集")
    
    # 2. 回転データの品質
    if 'rotation_sweep' in loaded_data:
        rotation_stats = analyze_rotation_data(loaded_data['rotation_sweep'])
        if rotation_stats['correlation'] > 0.95:
            quality_score += 1
            print("✓ 回転データ: 高品質（相関 > 0.95）")
        else:
            print("⚠ 回転データ: 品質要改善")
    else:
        print("✗ 回転データ: 未収集")
    
    # 3. ノイズデータの品質
    if 'noise_characterization' in loaded_data:
        quality_score += 1
        print("✓ ノイズデータ: 収集済み")
    else:
        print("✗ ノイズデータ: 未収集")
    
    # 4. VQEデータの品質
    if 'vqe_landscape' in loaded_data:
        vqe_data = loaded_data['vqe_landscape']
        if len(vqe_data) >= 100:  # 十分なデータポイント
            quality_score += 1
            print("✓ VQEデータ: 十分なデータポイント")
        else:
            print("⚠ VQEデータ: データポイント不足")
    else:
        print("✗ VQEデータ: 未収集")
    
    # 総合評価
    print(f"\n総合品質スコア: {quality_score}/{max_score}")
    
    if quality_score == max_score:
        print("🎯 CQT複素数解析に進む準備完了！")
        return True
    elif quality_score >= 2:
        print("⚠ 部分的にCQT解析可能、一部データの改善推奨")
        return True
    else:
        print("✗ データ品質不足、再収集が必要")
        return False

def visualize_data_overview(loaded_data):
    """収集データの概要を可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Bell状態の測定分布
    if 'bell_states' in loaded_data:
        bell_data = loaded_data['bell_states']
        ax = axes[0, 0]
        
        # 各Bell状態の測定数をプロット
        states = bell_data['state'].tolist()
        shots = bell_data['shots'].tolist()
        
        ax.bar(states, shots, alpha=0.7, color=['blue', 'red', 'green', 'orange'])
        ax.set_title('Bell States Measurement Count')
        ax.set_ylabel('Number of Shots')
        ax.tick_params(axis='x', rotation=45)
    
    # 2. 回転スイープの理論 vs 実測
    if 'rotation_sweep' in loaded_data:
        rotation_data = loaded_data['rotation_sweep']
        ax = axes[0, 1]
        
        ax.plot(rotation_data['angle'], rotation_data['theoretical_prob_1'], 'r--', 
                alpha=0.8, label='Theoretical')
        ax.plot(rotation_data['angle'], rotation_data['probability_1'], 'b-', 
                alpha=0.7, label='Measured')
        ax.set_title('Rotation Sweep: Theory vs Measurement')
        ax.set_xlabel('Angle (rad)')
        ax.set_ylabel('Probability |1⟩')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. ノイズ vs 忠実度
    if 'noise_characterization' in loaded_data:
        noise_data = loaded_data['noise_characterization']
        ax = axes[1, 0]
        
        ax.plot(noise_data['noise_level'], noise_data['fidelity'], 'ro-', alpha=0.7)
        ax.set_title('Noise Level vs Fidelity')
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Fidelity')
        ax.grid(True, alpha=0.3)
    
    # 4. VQEエネルギーランドスケープ（サンプル）
    if 'vqe_landscape' in loaded_data:
        vqe_data = loaded_data['vqe_landscape']
        ax = axes[1, 1]
        
        # 2Dヒートマップ用のデータ準備
        theta_unique = sorted(vqe_data['theta'].unique())
        phi_unique = sorted(vqe_data['phi'].unique())
        
        if len(theta_unique) > 1 and len(phi_unique) > 1:
            energy_matrix = np.zeros((len(phi_unique), len(theta_unique)))
            
            for i, phi in enumerate(phi_unique):
                for j, theta in enumerate(theta_unique):
                    energy_val = vqe_data[(vqe_data['theta'] == theta) & 
                                         (vqe_data['phi'] == phi)]['energy']
                    if not energy_val.empty:
                        energy_matrix[i, j] = energy_val.iloc[0]
            
            im = ax.imshow(energy_matrix, aspect='auto', origin='lower')
            ax.set_title('VQE Energy Landscape')
            ax.set_xlabel('Theta Index')
            ax.set_ylabel('Phi Index')
            plt.colorbar(im, ax=ax, label='Energy')
    
    plt.tight_layout()
    plt.savefig('collected_data/analysis_overview.png', dpi=150, bbox_inches='tight')
    print("\n可視化結果を collected_data/analysis_overview.png に保存しました")
    plt.show()

def main():
    """メイン解析実行"""
    print("=== CQT Theory - 収集データ解析開始 ===")
    
    # データ読み込み
    loaded_data = load_collected_data()
    
    if not loaded_data:
        print("解析可能なデータが見つかりません。先にデータ収集を実行してください。")
        return
    
    # 各データセットの詳細解析
    if 'bell_states' in loaded_data:
        analyze_bell_data(loaded_data['bell_states'])
    
    if 'rotation_sweep' in loaded_data:
        analyze_rotation_data(loaded_data['rotation_sweep'])
    
    if 'noise_characterization' in loaded_data:
        analyze_noise_data(loaded_data['noise_characterization'])
    
    # CQT解析準備状況の評価
    ready_for_cqt = prepare_for_cqt_analysis(loaded_data)
    
    # 可視化
    visualize_data_overview(loaded_data)
    
    # 次のステップの提案
    print("\n=== 次のステップ ===")
    if ready_for_cqt:
        print("1. complex_analysis.py で複素数変換を実行")
        print("2. pattern_discovery.py でBell状態シグネチャを発見")
        print("3. CQT v3トラッカーとの統合テスト")
    else:
        print("1. データ品質の改善")
        print("2. 追加データの収集")
        print("3. ノイズモデルの調整")

if __name__ == "__main__":
    main()