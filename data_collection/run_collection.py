"""
データ収集を実行するメインスクリプト
CQTプロジェクト用にNumPyベースで実装
"""
from quantum_datasets import QuantumDataCollector
from fetch_public_datasets import PublicDatasetFetcher
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    """データ収集のメイン実行関数"""
    
    print("=== CQT Theory - 量子データ収集開始 ===")
    print("NumPy + matplotlib + pandas ベースの実装")
    
    # 1. 量子データコレクターの初期化
    collector = QuantumDataCollector(base_path='collected_data')
    
    # 2. Bell状態のデータ収集
    print("\n1. Bell状態データの収集...")
    try:
        bell_data = collector.collect_bell_state_data(shots=8192)
        print(f"   ✓ 収集完了: {len(bell_data)} Bell状態")
        
        # Bell状態データの詳細表示
        for idx, row in bell_data.iterrows():
            state = row['state']
            counts = eval(row['counts']) if isinstance(row['counts'], str) else row['counts']
            total = sum(counts.values())
            print(f"     {state}: {counts} (total: {total})")
            
    except Exception as e:
        print(f"   ✗ Bell状態データ収集エラー: {e}")
        bell_data = pd.DataFrame()
    
    # 3. パラメトリック回転データの収集
    print("\n2. パラメトリック回転データの収集...")
    try:
        rotation_data = collector.collect_parametric_rotation_data()
        print(f"   ✓ 収集完了: {len(rotation_data)} データポイント")
        
        # 理論値と実測値の比較
        theoretical_mean = rotation_data['theoretical_prob_1'].mean()
        measured_mean = rotation_data['probability_1'].mean()
        print(f"     理論平均確率: {theoretical_mean:.3f}")
        print(f"     実測平均確率: {measured_mean:.3f}")
        print(f"     誤差: {abs(theoretical_mean - measured_mean):.3f}")
        
    except Exception as e:
        print(f"   ✗ 回転データ収集エラー: {e}")
        rotation_data = pd.DataFrame()
    
    # 4. エンタングルメントデータの収集
    print("\n3. エンタングルメントデータの収集...")
    entanglement_data = []
    for n_qubits in [2, 3, 4]:
        try:
            entangle_data = collector.collect_entanglement_data(n_qubits=n_qubits)
            entanglement_data.append(entangle_data)
            print(f"   ✓ {n_qubits}量子ビット: 完了 ({len(entangle_data)} 状態)")
            
            # エンタングルメントの特性表示
            for idx, row in entangle_data.iterrows():
                state_type = row['state_type']
                counts = eval(row['counts']) if isinstance(row['counts'], str) else row['counts']
                total = sum(counts.values())
                
                # エンタングルメント度の簡易計算
                if state_type == 'GHZ':
                    all_zeros = counts.get('0' * n_qubits, 0)
                    all_ones = counts.get('1' * n_qubits, 0)
                    entanglement_score = (all_zeros + all_ones) / total
                    print(f"     {state_type}: エンタングルメント度 = {entanglement_score:.3f}")
                
        except Exception as e:
            print(f"   ✗ {n_qubits}量子ビットエラー: {e}")
    
    # 5. ノイズ特性データの収集
    print("\n4. ノイズ特性データの収集...")
    try:
        noise_data = collector.collect_noise_characterization_data()
        print(f"   ✓ 収集完了: {len(noise_data)} ノイズレベル")
        
        # ノイズの影響表示
        for idx, row in noise_data.iterrows():
            noise_level = row['noise_level']
            fidelity = row['fidelity']
            print(f"     ノイズ {noise_level:.2f}: 忠実度 = {fidelity:.3f}")
            
    except Exception as e:
        print(f"   ✗ ノイズデータ収集エラー: {e}")
        noise_data = pd.DataFrame()
    
    # 6. VQEランドスケープデータの収集
    print("\n5. VQEランドスケープデータの収集...")
    try:
        vqe_data = collector.collect_vqe_landscape_data()
        print(f"   ✓ 収集完了: {len(vqe_data)} データポイント")
        
        # エネルギー統計
        min_energy = vqe_data['energy'].min()
        max_energy = vqe_data['energy'].max()
        mean_energy = vqe_data['energy'].mean()
        print(f"     エネルギー範囲: [{min_energy:.3f}, {max_energy:.3f}]")
        print(f"     平均エネルギー: {mean_energy:.3f}")
        
    except Exception as e:
        print(f"   ✗ VQEデータ収集エラー: {e}")
        vqe_data = pd.DataFrame()
    
    # 7. 公開データセットの取得
    print("\n6. 公開データセットの取得...")
    try:
        fetcher = PublicDatasetFetcher(cache_dir='public_datasets')
        fetcher.save_all_datasets()
        print("   ✓ 公開データセット取得完了")
        
    except Exception as e:
        print(f"   ✗ 公開データセット取得エラー: {e}")
    
    print("\n=== すべてのデータ収集が完了しました ===")
    
    # 簡単な可視化とCQT統合分析
    if not rotation_data.empty and not noise_data.empty:
        print("\n7. データ可視化とCQT統合分析...")
        try:
            visualize_collected_data(rotation_data, noise_data)
            cqt_integration_analysis(bell_data, rotation_data, noise_data)
            print("   ✓ 可視化完了")
        except Exception as e:
            print(f"   ✗ 可視化エラー: {e}")
    
    print("\n=== データ収集レポート生成 ===")
    generate_collection_report(bell_data, rotation_data, noise_data, vqe_data)

def visualize_collected_data(rotation_data: pd.DataFrame, noise_data: pd.DataFrame):
    """収集したデータの可視化"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 回転データのプロット
    if not rotation_data.empty:
        ax1.plot(rotation_data['angle'], rotation_data['probability_1'], 'b-o', alpha=0.7, label='実測')
        ax1.plot(rotation_data['angle'], rotation_data['theoretical_prob_1'], 'r--', alpha=0.7, label='理論')
        ax1.set_xlabel('Rotation Angle (rad)')
        ax1.set_ylabel('Probability |1⟩')
        ax1.set_title('単一量子ビット回転スイープ')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. ノイズデータのプロット
    if not noise_data.empty:
        ax2.plot(noise_data['noise_level'], noise_data['fidelity'], 'r-s', alpha=0.7)
        ax2.set_xlabel('ノイズレベル')
        ax2.set_ylabel('忠実度')
        ax2.set_title('Bell状態へのノイズ影響')
        ax2.grid(True, alpha=0.3)
    
    # 3. 測定統計のヒストグラム
    if not rotation_data.empty:
        residuals = rotation_data['probability_1'] - rotation_data['theoretical_prob_1']
        ax3.hist(residuals, bins=15, alpha=0.7, color='green')
        ax3.set_xlabel('測定誤差（実測 - 理論）')
        ax3.set_ylabel('頻度')
        ax3.set_title('測定誤差分布')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax3.grid(True, alpha=0.3)
    
    # 4. ノイズと忠実度の関係（対数プロット）
    if not noise_data.empty:
        ax4.semilogy(noise_data['noise_level'], 1 - noise_data['fidelity'], 'purple', marker='o')
        ax4.set_xlabel('ノイズレベル')
        ax4.set_ylabel('エラー率（対数）')
        ax4.set_title('ノイズ-エラー関係（対数スケール）')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('CQT Theory - 量子データ収集結果', fontsize=16)
    plt.tight_layout()
    
    # 保存
    os.makedirs('collected_data', exist_ok=True)
    plt.savefig('collected_data/data_collection_overview.png', dpi=150, bbox_inches='tight')
    plt.show()

def cqt_integration_analysis(bell_data: pd.DataFrame, rotation_data: pd.DataFrame, noise_data: pd.DataFrame):
    """CQT理論との統合分析"""
    print("\n--- CQT統合分析結果 ---")
    
    # 1. 複素軌跡への変換適合性
    if not rotation_data.empty:
        # 測定の一貫性評価
        consistency_score = 1.0 - np.std(rotation_data['probability_1'] - rotation_data['theoretical_prob_1'])
        print(f"1. 測定一貫性スコア: {consistency_score:.3f}")
        print(f"   → CQT複素表現への適合度: {'高' if consistency_score > 0.95 else '中' if consistency_score > 0.9 else '低'}")
    
    # 2. Bell状態での相関パターン
    if not bell_data.empty:
        print(f"2. Bell状態データ品質:")
        for idx, row in bell_data.iterrows():
            state = row['state']
            counts = eval(row['counts']) if isinstance(row['counts'], str) else row['counts']
            total = sum(counts.values())
            
            # 理想的なBell状態での期待値
            if state in ['phi_plus', 'phi_minus']:
                expected_states = ['00', '11']
            else:
                expected_states = ['01', '10']
            
            ideal_count = sum(counts.get(s, 0) for s in expected_states)
            fidelity = ideal_count / total if total > 0 else 0
            print(f"   {state}: 忠実度 = {fidelity:.3f}")
    
    # 3. ノイズ環境でのCQT検出能力
    if not noise_data.empty:
        print(f"3. CQTノイズ検出能力:")
        detectable_noise_threshold = None
        for idx, row in noise_data.iterrows():
            if row['fidelity'] < 0.9:  # 検出可能な閾値
                detectable_noise_threshold = row['noise_level']
                break
        
        if detectable_noise_threshold:
            print(f"   検出可能ノイズ閾値: {detectable_noise_threshold:.3f}")
            print(f"   → CQTエラー検出感度: {'高感度' if detectable_noise_threshold < 0.02 else '中感度'}")
        else:
            print("   → 全ノイズレベルで高忠実度維持")

def generate_collection_report(bell_data: pd.DataFrame, rotation_data: pd.DataFrame, 
                             noise_data: pd.DataFrame, vqe_data: pd.DataFrame):
    """データ収集レポートを生成"""
    report = []
    report.append("# CQT Theory - データ収集レポート")
    report.append(f"生成日時: {pd.Timestamp.now()}")
    report.append("")
    
    # データセット概要
    report.append("## 収集データセット概要")
    
    if not bell_data.empty:
        report.append(f"- Bell状態データ: {len(bell_data)} 状態")
        total_bell_measurements = 0
        for _, row in bell_data.iterrows():
            counts = eval(row['counts']) if isinstance(row['counts'], str) else row['counts']
            total_bell_measurements += sum(counts.values())
        report.append(f"  総測定数: {total_bell_measurements}")
    
    if not rotation_data.empty:
        report.append(f"- 回転スイープデータ: {len(rotation_data)} ポイント")
        report.append(f"  角度範囲: {rotation_data['angle'].min():.2f} - {rotation_data['angle'].max():.2f} rad")
    
    if not noise_data.empty:
        report.append(f"- ノイズ特性データ: {len(noise_data)} レベル")
        report.append(f"  ノイズ範囲: {noise_data['noise_level'].min():.3f} - {noise_data['noise_level'].max():.3f}")
    
    if not vqe_data.empty:
        report.append(f"- VQEランドスケープ: {len(vqe_data)} ポイント")
        report.append(f"  エネルギー範囲: {vqe_data['energy'].min():.3f} - {vqe_data['energy'].max():.3f}")
    
    report.append("")
    
    # CQT理論への適用性
    report.append("## CQT理論への適用性評価")
    
    if not rotation_data.empty:
        measurement_error = np.mean(np.abs(rotation_data['probability_1'] - rotation_data['theoretical_prob_1']))
        report.append(f"- 測定精度: 平均誤差 {measurement_error:.4f}")
        report.append(f"- 複素軌跡変換適合性: {'良好' if measurement_error < 0.05 else '要改善'}")
    
    if not noise_data.empty:
        min_detectable_noise = noise_data[noise_data['fidelity'] < 0.95]['noise_level'].min()
        if pd.notna(min_detectable_noise):
            report.append(f"- ノイズ検出感度: {min_detectable_noise:.3f} レベルから検出可能")
        else:
            report.append("- ノイズ検出感度: 高レベルノイズまで安定")
    
    report.append("")
    report.append("## 推奨事項")
    report.append("1. 収集データをCQT v3トラッカーでの複素軌跡変換に使用")
    report.append("2. Bell状態データを基準としたエラー検出閾値調整")
    report.append("3. ノイズモデルの更なる精緻化")
    
    # レポート保存
    os.makedirs('collected_data', exist_ok=True)
    with open('collected_data/collection_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("   ✓ レポート生成完了: collected_data/collection_report.md")

if __name__ == "__main__":
    main()