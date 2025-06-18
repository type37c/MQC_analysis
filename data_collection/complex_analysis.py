"""
CQT理論による複素数解析
収集した量子測定データを複素数軌跡に変換し、パターンを発見する
"""
import json
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import sys
import os

# CQTトラッカーをインポート
sys.path.append('../src')
try:
    from cqt_tracker_v3 import OptimizedCQTTracker, MeasurementRecord
except ImportError:
    print("警告: CQT v3トラッカーが見つかりません。基本的な複素数変換のみ実行します。")
    OptimizedCQTTracker = None

class CQTComplexAnalyzer:
    """CQT理論による複素数解析クラス"""
    
    def __init__(self):
        self.bell_signatures = {}
        self.complex_trajectories = {}
        
    def parse_counts_string(self, counts_str: str) -> Dict[str, int]:
        """counts文字列を辞書に変換"""
        try:
            # NumPy型の文字列表現を処理
            counts_str = counts_str.replace("np.str_('", "'").replace("np.int64(", "").replace("')", "'").replace(")", "")
            return ast.literal_eval(counts_str)
        except Exception as e:
            print(f"counts文字列解析エラー: {e}")
            return {}
    
    def bell_state_to_complex_trajectory(self, bell_data_row: pd.Series) -> List[complex]:
        """
        Bell状態の測定結果を複素数軌跡として解析
        
        CQT理論では：
        - 実部: 相関測定 (correlation measurement)
        - 虚部: 不確実性 (uncertainty)
        """
        state = bell_data_row['state']
        counts_str = bell_data_row['counts']
        shots = bell_data_row['shots']
        
        # counts文字列を辞書に変換
        counts = self.parse_counts_string(counts_str)
        if not counts:
            return []
        
        # 測定軌跡を生成（累積測定をシミュレート）
        trajectory = []
        total_counts = sum(counts.values())
        
        # 100ステップで軌跡を作成
        for step in range(1, 101):
            # 累積測定数
            progress = step / 100
            
            # 各状態の累積カウント
            accumulated_counts = {}
            for outcome, count in counts.items():
                accumulated_counts[outcome] = int(count * progress)
            
            # CQT複素数変換
            z = self._convert_to_cqt_complex(accumulated_counts, state, progress)
            trajectory.append(z)
        
        return trajectory
    
    def _convert_to_cqt_complex(self, counts: Dict[str, int], state_type: str, progress: float) -> complex:
        """
        測定結果をCQT複素数に変換
        
        Args:
            counts: 測定カウント {'00': n1, '11': n2, ...}
            state_type: Bell状態の種類
            progress: 測定進行度 (0-1)
        
        Returns:
            complex: z = correlation + i*uncertainty
        """
        total = sum(counts.values())
        if total == 0:
            return complex(0, 0)
        
        # Bell状態別の期待値パターン
        if state_type in ['phi_plus', 'phi_minus']:
            # |Φ±⟩ = (|00⟩ ± |11⟩)/√2
            correlated_count = counts.get('00', 0) + counts.get('11', 0)
            anti_correlated_count = counts.get('01', 0) + counts.get('10', 0)
            
        elif state_type in ['psi_plus', 'psi_minus']:
            # |Ψ±⟩ = (|01⟩ ± |10⟩)/√2
            correlated_count = counts.get('01', 0) + counts.get('10', 0)
            anti_correlated_count = counts.get('00', 0) + counts.get('11', 0)
        
        else:
            # 未知の状態
            correlated_count = 0
            anti_correlated_count = total
        
        # 相関度（実部）: -1（完全反相関） to +1（完全相関）
        correlation = (correlated_count - anti_correlated_count) / total if total > 0 else 0
        
        # 不確実性（虚部）: 測定の統計的不確実性
        if total > 0:
            # 統計的不確実性 = √(N_corr * N_anti) / N_total
            uncertainty = np.sqrt(correlated_count * anti_correlated_count) / total if correlated_count > 0 and anti_correlated_count > 0 else 0
            
            # 進行に伴う不確実性の減少
            uncertainty *= np.sqrt((1 - progress) + 0.1)  # 最低限の不確実性を保持
        else:
            uncertainty = 1.0  # 測定前は最大不確実性
        
        return complex(correlation, uncertainty)
    
    def analyze_rotation_trajectory(self, rotation_data: pd.DataFrame) -> List[complex]:
        """回転スイープデータの複素数軌跡解析"""
        trajectory = []
        
        for _, row in rotation_data.iterrows():
            angle = row['angle']
            prob_1 = row['probability_1']
            theoretical_prob = row['theoretical_prob_1']
            
            # 測定の方向性（実部）
            direction = prob_1 - 0.5  # -0.5 to +0.5 の範囲に正規化
            
            # 測定の不確実性（虚部）
            deviation = abs(prob_1 - theoretical_prob)
            uncertainty = deviation + 0.01  # 最小不確実性
            
            z = complex(direction, uncertainty)
            trajectory.append(z)
        
        return trajectory
    
    def detect_cqt_patterns(self, trajectory: List[complex]) -> Dict[str, float]:
        """
        複素数軌跡からCQTパターンを検出
        
        Returns:
            Dict: パターン解析結果
        """
        if not trajectory:
            return {}
        
        real_parts = [z.real for z in trajectory]
        imag_parts = [z.imag for z in trajectory]
        
        # パターン特徴量
        patterns = {
            'final_correlation': real_parts[-1],
            'final_uncertainty': imag_parts[-1],
            'correlation_stability': 1.0 - np.std(real_parts),
            'uncertainty_evolution': imag_parts[0] - imag_parts[-1],  # 不確実性の減少
            'trajectory_complexity': np.sum(np.abs(np.diff([abs(z) for z in trajectory]))),
            'convergence_rate': self._calculate_convergence_rate(trajectory),
            'spiral_tendency': self._detect_spiral_pattern(trajectory),
            'symmetry_score': self._calculate_symmetry(trajectory)
        }
        
        return patterns
    
    def _calculate_convergence_rate(self, trajectory: List[complex]) -> float:
        """軌跡の収束率を計算"""
        if len(trajectory) < 10:
            return 0.0
        
        # 最後の10%の軌跡の分散を計算
        final_segment = trajectory[-10:]
        final_variance = np.var([abs(z) for z in final_segment])
        
        # 最初の10%の軌跡の分散
        initial_segment = trajectory[:10]
        initial_variance = np.var([abs(z) for z in initial_segment])
        
        # 収束率 = 分散の減少率
        if initial_variance > 0:
            return 1.0 - (final_variance / initial_variance)
        else:
            return 1.0
    
    def _detect_spiral_pattern(self, trajectory: List[complex]) -> float:
        """軌跡のスパイラルパターンを検出"""
        if len(trajectory) < 3:
            return 0.0
        
        # 位相角の変化を追跡
        angles = [np.angle(z) for z in trajectory if abs(z) > 1e-10]
        
        if len(angles) < 3:
            return 0.0
        
        # 角度変化の一貫性
        angle_diffs = np.diff(angles)
        
        # 2πの境界を考慮した角度差の正規化
        angle_diffs = [(diff + np.pi) % (2 * np.pi) - np.pi for diff in angle_diffs]
        
        # スパイラル傾向 = 角度変化の一貫性
        if len(angle_diffs) > 1:
            spiral_score = 1.0 - np.std(angle_diffs) / np.pi
            return max(0, spiral_score)
        else:
            return 0.0
    
    def _calculate_symmetry(self, trajectory: List[complex]) -> float:
        """軌跡の対称性を計算"""
        if len(trajectory) < 4:
            return 0.0
        
        # 軌跡を前半と後半に分割
        mid = len(trajectory) // 2
        first_half = trajectory[:mid]
        second_half = trajectory[mid:]
        
        # 後半を反転
        reversed_second = second_half[::-1]
        
        # 対称性スコア計算
        min_len = min(len(first_half), len(reversed_second))
        if min_len == 0:
            return 0.0
        
        symmetry_errors = []
        for i in range(min_len):
            error = abs(first_half[i] - reversed_second[i])
            symmetry_errors.append(error)
        
        # 対称性スコア = 1 - 平均誤差
        mean_error = np.mean(symmetry_errors)
        symmetry_score = max(0, 1.0 - mean_error)
        
        return symmetry_score
    
    def analyze_all_bell_states(self, bell_data: pd.DataFrame) -> Dict[str, Dict]:
        """全Bell状態の複素数解析"""
        results = {}
        
        print("=== Bell状態の複素数軌跡解析 ===")
        
        for idx, row in bell_data.iterrows():
            state = row['state']
            print(f"\n{state} の解析中...")
            
            # 複素数軌跡の生成
            trajectory = self.bell_state_to_complex_trajectory(row)
            
            if trajectory:
                # パターン検出
                patterns = self.detect_cqt_patterns(trajectory)
                
                # 結果保存
                results[state] = {
                    'trajectory': trajectory,
                    'patterns': patterns,
                    'final_position': trajectory[-1] if trajectory else complex(0, 0)
                }
                
                # 結果表示
                print(f"  最終位置: {trajectory[-1]:.4f}")
                print(f"  相関度: {patterns['final_correlation']:.4f}")
                print(f"  不確実性: {patterns['final_uncertainty']:.4f}")
                print(f"  収束率: {patterns['convergence_rate']:.4f}")
                print(f"  対称性: {patterns['symmetry_score']:.4f}")
            else:
                print(f"  軌跡生成に失敗")
                results[state] = {'trajectory': [], 'patterns': {}, 'final_position': complex(0, 0)}
        
        self.complex_trajectories = results
        return results
    
    def compare_with_cqt_v3(self, bell_data: pd.DataFrame):
        """CQT v3トラッカーとの比較解析"""
        if OptimizedCQTTracker is None:
            print("CQT v3トラッカーが利用できません。基本解析のみ実行します。")
            return
        
        print("\n=== CQT v3トラッカーとの比較 ===")
        
        for idx, row in bell_data.iterrows():
            state = row['state']
            counts = self.parse_counts_string(row['counts'])
            
            print(f"\n{state}:")
            
            # v3トラッカーでの解析
            tracker = OptimizedCQTTracker(system_dim=2)
            
            # 測定データをv3に入力
            for outcome_str, count in counts.items():
                outcome = int(outcome_str[0])  # 最初のビットを使用
                
                for _ in range(count):
                    complex_val = tracker.add_measurement(outcome)
            
            # v3の結果
            v3_trajectory = tracker.trajectory
            v3_analysis = tracker.analyze_trajectory()
            
            print(f"  v3最終軌跡長: {len(v3_trajectory)}")
            if v3_trajectory:
                print(f"  v3最終位置: {v3_trajectory[-1]:.4f}")
            
            if v3_analysis:
                print(f"  v3解析結果: {v3_analysis}")
            
            # 本解析との比較
            if state in self.complex_trajectories:
                our_final = self.complex_trajectories[state]['final_position']
                print(f"  本解析最終位置: {our_final:.4f}")
                
                if v3_trajectory:
                    difference = abs(our_final - v3_trajectory[-1])
                    print(f"  位置差: {difference:.4f}")
    
    def visualize_complex_trajectories(self, results: Dict[str, Dict]):
        """複素数軌跡の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (state, data) in enumerate(results.items()):
            if i >= 4:
                break
                
            ax = axes[i]
            trajectory = data['trajectory']
            
            if trajectory:
                real_parts = [z.real for z in trajectory]
                imag_parts = [z.imag for z in trajectory]
                
                # 軌跡のプロット
                ax.plot(real_parts, imag_parts, color=colors[i], alpha=0.7, linewidth=2, label=f'{state} trajectory')
                
                # 開始点と終了点
                ax.scatter(real_parts[0], imag_parts[0], color=colors[i], s=100, marker='o', label='Start', alpha=0.8)
                ax.scatter(real_parts[-1], imag_parts[-1], color=colors[i], s=150, marker='*', label='End', alpha=1.0)
                
                # 軌跡の方向を示す矢印
                if len(trajectory) > 10:
                    for j in range(0, len(trajectory)-1, len(trajectory)//5):
                        dx = real_parts[j+1] - real_parts[j]
                        dy = imag_parts[j+1] - imag_parts[j]
                        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                            ax.arrow(real_parts[j], imag_parts[j], dx*0.5, dy*0.5, 
                                   head_width=0.02, head_length=0.02, fc=colors[i], ec=colors[i], alpha=0.5)
                
                ax.set_title(f'{state} Complex Trajectory')
                ax.set_xlabel('Real (Correlation)')
                ax.set_ylabel('Imaginary (Uncertainty)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # 軌跡の特徴を注釈
                patterns = data['patterns']
                if patterns:
                    ax.text(0.05, 0.95, f"Convergence: {patterns.get('convergence_rate', 0):.3f}\n"
                                       f"Symmetry: {patterns.get('symmetry_score', 0):.3f}", 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No trajectory data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{state} (No Data)')
        
        plt.suptitle('CQT Theory - Bell States Complex Trajectories', fontsize=16)
        plt.tight_layout()
        plt.savefig('collected_data/complex_trajectories.png', dpi=150, bbox_inches='tight')
        print("\n複素数軌跡の可視化を collected_data/complex_trajectories.png に保存しました")
        plt.show()
    
    def generate_cqt_signatures(self, results: Dict[str, Dict]) -> Dict[str, Dict]:
        """各Bell状態のCQTシグネチャを生成"""
        signatures = {}
        
        print("\n=== CQT Bell状態シグネチャ生成 ===")
        
        for state, data in results.items():
            trajectory = data['trajectory']
            patterns = data['patterns']
            
            if trajectory and patterns:
                signature = {
                    'complex_signature': data['final_position'],
                    'correlation_range': [min([z.real for z in trajectory]), max([z.real for z in trajectory])],
                    'uncertainty_range': [min([z.imag for z in trajectory]), max([z.imag for z in trajectory])],
                    'trajectory_characteristics': {
                        'convergence_rate': patterns.get('convergence_rate', 0),
                        'symmetry_score': patterns.get('symmetry_score', 0),
                        'spiral_tendency': patterns.get('spiral_tendency', 0),
                        'complexity': patterns.get('trajectory_complexity', 0)
                    },
                    'classification_features': [
                        patterns.get('final_correlation', 0),
                        patterns.get('final_uncertainty', 0),
                        patterns.get('convergence_rate', 0),
                        patterns.get('symmetry_score', 0)
                    ]
                }
                
                signatures[state] = signature
                
                print(f"\n{state}:")
                print(f"  複素シグネチャ: {signature['complex_signature']:.4f}")
                print(f"  相関範囲: [{signature['correlation_range'][0]:.3f}, {signature['correlation_range'][1]:.3f}]")
                print(f"  不確実性範囲: [{signature['uncertainty_range'][0]:.3f}, {signature['uncertainty_range'][1]:.3f}]")
                print(f"  収束率: {signature['trajectory_characteristics']['convergence_rate']:.3f}")
        
        # シグネチャをJSONで保存
        signature_file = 'collected_data/cqt_bell_signatures.json'
        with open(signature_file, 'w') as f:
            # complex型は直接JSONシリアライズできないので変換
            serializable_sigs = {}
            for state, sig in signatures.items():
                serializable_sigs[state] = {
                    'complex_signature_real': sig['complex_signature'].real,
                    'complex_signature_imag': sig['complex_signature'].imag,
                    'correlation_range': sig['correlation_range'],
                    'uncertainty_range': sig['uncertainty_range'],
                    'trajectory_characteristics': sig['trajectory_characteristics'],
                    'classification_features': sig['classification_features']
                }
            
            json.dump(serializable_sigs, f, indent=2)
        
        print(f"\nCQTシグネチャを {signature_file} に保存しました")
        
        return signatures

def main():
    """メイン実行関数"""
    print("=== CQT Theory - 複素数解析開始 ===")
    
    # データ読み込み
    bell_data_file = 'collected_data/bell_states/bell_measurement_data.csv'
    
    if not os.path.exists(bell_data_file):
        print("Bell状態データが見つかりません。先にrun_collection.pyを実行してください。")
        return
    
    bell_data = pd.read_csv(bell_data_file)
    
    # CQT複素数解析の実行
    analyzer = CQTComplexAnalyzer()
    
    # 全Bell状態の解析
    results = analyzer.analyze_all_bell_states(bell_data)
    
    # CQT v3との比較
    analyzer.compare_with_cqt_v3(bell_data)
    
    # 可視化
    analyzer.visualize_complex_trajectories(results)
    
    # CQTシグネチャ生成
    signatures = analyzer.generate_cqt_signatures(results)
    
    print("\n=== 複素数解析完了 ===")
    print("次のステップ: pattern_discovery.py でパターン発見を実行")

if __name__ == "__main__":
    main()