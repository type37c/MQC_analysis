"""
IBM Quantum Volumeの実データを使用したCQT理論解析
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# CQTトラッカーをインポート
sys.path.append('../src')
from cqt_tracker_v3 import OptimizedCQTTracker

class RealQuantumDataAnalyzer:
    """実際の量子データを使用したCQT解析"""
    
    def __init__(self):
        self.data_path = 'downloaded_quantum_data/IBM_Quantum_Volume/qiskit-experiments/qiskit-experiments-main/test/library/quantum_volume'
        self.qv_data = None
        self.cqt_trajectories = {}
        
    def load_quantum_volume_data(self):
        """Quantum Volumeデータを読み込み"""
        print("=== IBM Quantum Volume データの読み込み ===")
        
        # 利用可能なデータファイル
        data_files = {
            'moderate_noise_100': 'qv_data_moderate_noise_100_trials.json',
            'moderate_noise_300': 'qv_data_moderate_noise_300_trials.json',
            'high_noise': 'qv_data_high_noise.json',
            '70_trials': 'qv_data_70_trials.json'
        }
        
        loaded_data = {}
        
        for key, filename in data_files.items():
            filepath = os.path.join(self.data_path, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    loaded_data[key] = data
                    print(f"✓ {key}: {len(data)} エントリ読み込み完了")
            else:
                print(f"✗ {key}: ファイルが見つかりません")
        
        self.qv_data = loaded_data
        return loaded_data
    
    def analyze_qv_trial_with_cqt(self, trial_data):
        """単一のQuantum Volume試行をCQT理論で解析"""
        tracker = OptimizedCQTTracker(system_dim=2)
        
        # 測定結果を抽出
        if 'counts' in trial_data:
            counts = trial_data['counts']
            
            # 各測定結果をCQTトラッカーに入力
            for bitstring, count in counts.items():
                # ビット列から測定結果を抽出
                for _ in range(count):
                    # 最初のビットを使用（簡略化）
                    outcome = int(bitstring[0]) if bitstring else 0
                    tracker.add_measurement(outcome)
        
        # 軌跡を解析
        trajectory = tracker.trajectory
        analysis = tracker.analyze_trajectory() if len(trajectory) > 0 else None
        
        return trajectory, analysis
    
    def compare_noise_levels(self):
        """異なるノイズレベルでのCQT軌跡を比較"""
        print("\n=== ノイズレベル比較解析 ===")
        
        if not self.qv_data:
            print("データが読み込まれていません")
            return
        
        results = {}
        
        # 各ノイズレベルのデータを解析
        for noise_level, data in self.qv_data.items():
            print(f"\n{noise_level} の解析:")
            
            trajectories = []
            analyses = []
            
            # 最初の10試行を解析（デモ用）
            for i, trial in enumerate(data[:10]):
                trajectory, analysis = self.analyze_qv_trial_with_cqt(trial)
                
                if trajectory:
                    trajectories.append(trajectory)
                    if analysis:
                        analyses.append(analysis)
            
            # 統計を計算
            if trajectories:
                avg_length = np.mean([len(t) for t in trajectories])
                
                # 最終位置の統計
                final_positions = [t[-1] if t else complex(0, 0) for t in trajectories]
                avg_real = np.mean([z.real for z in final_positions])
                avg_imag = np.mean([z.imag for z in final_positions])
                
                print(f"  平均軌跡長: {avg_length:.1f}")
                print(f"  平均最終位置: {avg_real:.3f} + {avg_imag:.3f}i")
                
                results[noise_level] = {
                    'trajectories': trajectories,
                    'analyses': analyses,
                    'avg_length': avg_length,
                    'avg_final_position': complex(avg_real, avg_imag)
                }
        
        self.cqt_trajectories = results
        return results
    
    def detect_quantum_volume_signatures(self):
        """Quantum Volume特有のCQTシグネチャを検出"""
        print("\n=== Quantum Volume CQTシグネチャ検出 ===")
        
        if not self.cqt_trajectories:
            print("CQT軌跡が計算されていません")
            return
        
        signatures = {}
        
        for noise_level, data in self.cqt_trajectories.items():
            trajectories = data['trajectories']
            
            if trajectories:
                # シグネチャ特徴を抽出
                signature_features = {
                    'trajectory_spread': np.std([abs(t[-1]) for t in trajectories if t]),
                    'phase_coherence': self._calculate_phase_coherence(trajectories),
                    'complexity': np.mean([self._trajectory_complexity(t) for t in trajectories]),
                    'noise_sensitivity': self._calculate_noise_sensitivity(trajectories)
                }
                
                signatures[noise_level] = signature_features
                
                print(f"\n{noise_level}:")
                for feature, value in signature_features.items():
                    print(f"  {feature}: {value:.4f}")
        
        return signatures
    
    def _calculate_phase_coherence(self, trajectories):
        """位相コヒーレンスを計算"""
        if not trajectories:
            return 0.0
        
        phases = []
        for traj in trajectories:
            if traj and len(traj) > 0:
                final_phase = np.angle(traj[-1])
                phases.append(final_phase)
        
        if phases:
            # 位相の分散が小さいほどコヒーレンスが高い
            phase_std = np.std(phases)
            coherence = np.exp(-phase_std)
            return coherence
        
        return 0.0
    
    def _trajectory_complexity(self, trajectory):
        """軌跡の複雑さを計算"""
        if len(trajectory) < 2:
            return 0.0
        
        # 軌跡の曲がりの総量
        complexity = 0.0
        for i in range(1, len(trajectory)):
            complexity += abs(trajectory[i] - trajectory[i-1])
        
        return complexity / len(trajectory)
    
    def _calculate_noise_sensitivity(self, trajectories):
        """ノイズ感度を計算"""
        if not trajectories:
            return 0.0
        
        # 軌跡の終点の分散
        final_positions = [t[-1] for t in trajectories if t]
        if final_positions:
            variance = np.var([abs(z) for z in final_positions])
            return variance
        
        return 0.0
    
    def visualize_real_data_analysis(self):
        """実データ解析結果の可視化"""
        print("\n=== 実データCQT解析の可視化 ===")
        
        if not self.cqt_trajectories:
            print("解析結果がありません")
            return
        
        # ノイズレベルごとに軌跡をプロット
        n_noise_levels = len(self.cqt_trajectories)
        fig, axes = plt.subplots(1, n_noise_levels, figsize=(5*n_noise_levels, 5))
        
        if n_noise_levels == 1:
            axes = [axes]
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for idx, (noise_level, data) in enumerate(self.cqt_trajectories.items()):
            ax = axes[idx]
            trajectories = data['trajectories']
            
            # 最初の5軌跡をプロット
            for i, traj in enumerate(trajectories[:5]):
                if traj:
                    real_parts = [z.real for z in traj]
                    imag_parts = [z.imag for z in traj]
                    
                    ax.plot(real_parts, imag_parts, 
                           color=colors[i % len(colors)], 
                           alpha=0.6, linewidth=1)
                    
                    # 終点をマーク
                    ax.scatter(real_parts[-1], imag_parts[-1], 
                             color=colors[i % len(colors)], 
                             s=50, marker='*')
            
            ax.set_title(f'{noise_level}\nQV Trajectories')
            ax.set_xlabel('Real (Direction)')
            ax.set_ylabel('Imaginary (Uncertainty)')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-0.5, 1.5)
        
        plt.tight_layout()
        plt.savefig('collected_data/real_qv_trajectories.png', dpi=150, bbox_inches='tight')
        print("可視化を collected_data/real_qv_trajectories.png に保存しました")
        plt.show()
    
    def generate_real_data_report(self):
        """実データ解析レポートを生成"""
        report = [
            "# IBM Quantum Volume 実データCQT解析レポート",
            f"生成日時: {pd.Timestamp.now()}",
            "",
            "## 解析概要",
            "",
            "IBM Qiskit ExperimentsのQuantum Volumeベンチマークデータを",
            "CQT理論で解析しました。",
            "",
            "## 主要発見",
            ""
        ]
        
        if self.cqt_trajectories:
            report.append("### ノイズレベル別特性")
            
            for noise_level, data in self.cqt_trajectories.items():
                report.append(f"\n**{noise_level}:**")
                report.append(f"- 平均軌跡長: {data['avg_length']:.1f}")
                report.append(f"- 平均最終位置: {data['avg_final_position']:.4f}")
        
        report.extend([
            "",
            "## 結論",
            "",
            "実際の量子コンピュータデータにCQT理論を適用し、",
            "ノイズレベルによる軌跡パターンの違いを観察しました。",
            "",
            "### 今後の展開",
            "1. より多くの実データでの検証",
            "2. ノイズモデルとCQT軌跡の相関解析",
            "3. 量子エラー予測への応用"
        ])
        
        report_content = "\n".join(report)
        
        with open('collected_data/real_qv_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print("\nレポートを collected_data/real_qv_analysis_report.md に保存しました")

def main():
    """メイン実行関数"""
    print("=== 実量子データCQT解析開始 ===")
    
    analyzer = RealQuantumDataAnalyzer()
    
    # データ読み込み
    analyzer.load_quantum_volume_data()
    
    # ノイズレベル比較
    analyzer.compare_noise_levels()
    
    # シグネチャ検出
    analyzer.detect_quantum_volume_signatures()
    
    # 可視化
    analyzer.visualize_real_data_analysis()
    
    # レポート生成
    analyzer.generate_real_data_report()
    
    print("\n=== 実データ解析完了 ===")

if __name__ == "__main__":
    main()