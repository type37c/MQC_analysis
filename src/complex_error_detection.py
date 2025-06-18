"""
複素演算によるエラー検出の高度化
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from complex_cqt_operations import ComplexCQTAnalyzer

class ComplexErrorDetector:
    """複素演算を用いた量子エラー検出"""
    
    def __init__(self, reference_trajectory):
        self.reference = np.array(reference_trajectory)
        self.analyzer = ComplexCQTAnalyzer(reference_trajectory)
        self.reference_features = self._extract_reference_features()
    
    def _extract_reference_features(self):
        """参照軌跡の特徴を抽出"""
        return {
            'fourier': self.analyzer.fourier_analysis(),
            'w_pattern': self.analyzer.analyze_w_pattern(),
            'invariants': self.analyzer.calculate_geometric_invariants()
        }
    
    def detect_errors(self, test_trajectory, window_size=20):
        """複素演算によるエラー検出"""
        test = np.array(test_trajectory)
        errors_detected = []
        
        for i in range(len(test) - window_size):
            window = test[i:i+window_size]
            
            # 1. 位相コヒーレンスの破れ
            phase_error = self._detect_phase_decoherence(window, i)
            if phase_error:
                errors_detected.append(phase_error)
            
            # 2. 振幅の異常
            amplitude_error = self._detect_amplitude_anomaly(window, i)
            if amplitude_error:
                errors_detected.append(amplitude_error)
            
            # 3. 軌跡の不連続性
            discontinuity = self._detect_trajectory_discontinuity(window, i)
            if discontinuity:
                errors_detected.append(discontinuity)
        
        return errors_detected
    
    def _detect_phase_decoherence(self, window, position):
        """位相のデコヒーレンスを検出"""
        # 位相の分散を計算
        phases = np.angle(window)
        phase_variance = np.var(np.unwrap(phases))
        
        # 参照軌跡の位相分散と比較
        ref_phases = np.angle(self.reference)
        ref_variance = np.var(np.unwrap(ref_phases))
        
        if phase_variance > 3 * ref_variance:  # 3σ基準
            return {
                'type': 'phase_decoherence',
                'position': position,
                'severity': phase_variance / ref_variance,
                'phase_variance': phase_variance
            }
        return None
    
    def _detect_amplitude_anomaly(self, window, position):
        """振幅の異常を検出"""
        amplitudes = np.abs(window)
        
        # 振幅の急激な変化
        amp_diff = np.abs(np.diff(amplitudes))
        if np.max(amp_diff) > 0.5:  # 閾値
            return {
                'type': 'amplitude_jump',
                'position': position + np.argmax(amp_diff),
                'severity': np.max(amp_diff),
                'amplitude_change': np.max(amp_diff)
            }
        return None
    
    def _detect_trajectory_discontinuity(self, window, position):
        """軌跡の不連続性を検出"""
        # 複素微分による速度計算
        velocity = np.diff(window)
        
        # 速度の急激な変化（加速度）
        acceleration = np.diff(velocity)
        
        if len(acceleration) > 0:
            max_acc = np.max(np.abs(acceleration))
            if max_acc > 1.0:  # 閾値
                return {
                    'type': 'trajectory_discontinuity',
                    'position': position + np.argmax(np.abs(acceleration)),
                    'severity': max_acc,
                    'acceleration': max_acc
                }
        return None
    
    def classify_error_type(self, error_info):
        """エラーの種類を分類"""
        if error_info['type'] == 'phase_decoherence':
            if error_info['severity'] > 5:
                return 'severe_decoherence'
            else:
                return 'mild_decoherence'
        
        elif error_info['type'] == 'amplitude_jump':
            if error_info['amplitude_change'] > 0.8:
                return 'bit_flip_error'
            else:
                return 'amplitude_damping'
        
        elif error_info['type'] == 'trajectory_discontinuity':
            return 'measurement_error'
        
        return 'unknown_error'
    
    def analyze_error_pattern(self, errors):
        """エラーパターンの詳細分析"""
        if not errors:
            return {'no_errors': True}
        
        # エラータイプ別の統計
        error_types = {}
        positions = []
        severities = []
        
        for error in errors:
            error_type = self.classify_error_type(error)
            error_types[error_type] = error_types.get(error_type, 0) + 1
            positions.append(error['position'])
            severities.append(error['severity'])
        
        # エラーの時間分布
        position_clusters = self._find_error_clusters(positions)
        
        # 深刻度の統計
        severity_stats = {
            'mean': np.mean(severities),
            'std': np.std(severities),
            'max': np.max(severities),
            'min': np.min(severities)
        }
        
        return {
            'total_errors': len(errors),
            'error_types': error_types,
            'position_clusters': position_clusters,
            'severity_stats': severity_stats,
            'error_rate': len(errors) / len(positions) if positions else 0
        }
    
    def _find_error_clusters(self, positions, cluster_threshold=10):
        """エラーの時間的クラスタリング"""
        if not positions:
            return []
        
        positions = sorted(positions)
        clusters = []
        current_cluster = [positions[0]]
        
        for i in range(1, len(positions)):
            if positions[i] - positions[i-1] <= cluster_threshold:
                current_cluster.append(positions[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [positions[i]]
        
        clusters.append(current_cluster)
        
        return [{'start': min(cluster), 'end': max(cluster), 'size': len(cluster)} 
                for cluster in clusters]
    
    def visualize_error_detection(self, test_trajectory, errors, save_path='error_detection.png'):
        """エラー検出結果の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        test = np.array(test_trajectory)
        
        # 1. 軌跡とエラー位置
        ax1 = axes[0, 0]
        ax1.plot(test.real, test.imag, 'b-', alpha=0.7, label='Trajectory')
        
        # エラー位置をマーク
        for error in errors:
            pos = error['position']
            if pos < len(test):
                ax1.scatter(test[pos].real, test[pos].imag, 
                           color='red', s=100, marker='x', alpha=0.8)
        
        ax1.set_xlabel('Real')
        ax1.set_ylabel('Imaginary')
        ax1.set_title('Trajectory with Error Positions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 時系列でのエラー検出
        ax2 = axes[0, 1]
        time_series = np.abs(test)
        ax2.plot(time_series, 'b-', alpha=0.7, label='Amplitude')
        
        for error in errors:
            pos = error['position']
            if pos < len(time_series):
                ax2.axvline(x=pos, color='red', alpha=0.6, linestyle='--')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Time Series with Error Markers')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. エラータイプ分布
        ax3 = axes[1, 0]
        error_analysis = self.analyze_error_pattern(errors)
        
        if 'error_types' in error_analysis:
            types = list(error_analysis['error_types'].keys())
            counts = list(error_analysis['error_types'].values())
            
            ax3.bar(types, counts, alpha=0.7, color=['red', 'orange', 'yellow', 'green'][:len(types)])
            ax3.set_xlabel('Error Type')
            ax3.set_ylabel('Count')
            ax3.set_title('Error Type Distribution')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. 深刻度の分布
        ax4 = axes[1, 1]
        if errors:
            severities = [error['severity'] for error in errors]
            ax4.hist(severities, bins=10, alpha=0.7, color='purple')
            ax4.set_xlabel('Severity')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Error Severity Distribution')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_error_report(self, test_trajectory, errors, save_path='error_report.txt'):
        """エラー検出の詳細レポート生成"""
        analysis = self.analyze_error_pattern(errors)
        
        report = f"""
=== 複素演算によるCQTエラー検出レポート ===

軌跡情報:
- 総測定点数: {len(test_trajectory)}
- 検出されたエラー数: {analysis.get('total_errors', 0)}
- エラー率: {analysis.get('error_rate', 0):.4f}

エラータイプ別統計:
"""
        
        if 'error_types' in analysis:
            for error_type, count in analysis['error_types'].items():
                report += f"- {error_type}: {count}件\n"
        
        report += f"""
深刻度統計:
- 平均: {analysis.get('severity_stats', {}).get('mean', 0):.3f}
- 標準偏差: {analysis.get('severity_stats', {}).get('std', 0):.3f}
- 最大: {analysis.get('severity_stats', {}).get('max', 0):.3f}
- 最小: {analysis.get('severity_stats', {}).get('min', 0):.3f}

エラークラスタ:
"""
        
        if 'position_clusters' in analysis:
            for i, cluster in enumerate(analysis['position_clusters']):
                report += f"- クラスタ{i+1}: 位置{cluster['start']}-{cluster['end']} ({cluster['size']}件)\n"
        
        report += f"""
検出された個別エラー:
"""
        
        for i, error in enumerate(errors[:10]):  # 最初の10件
            error_type = self.classify_error_type(error)
            report += f"- エラー{i+1}: 位置{error['position']}, タイプ: {error_type}, 深刻度: {error['severity']:.3f}\n"
        
        if len(errors) > 10:
            report += f"... 他{len(errors) - 10}件\n"
        
        report += "\n=== レポート終了 ===\n"
        
        # ファイルに保存
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        return report

def compute_complex_correlation(traj1, traj2):
    """2つの複素軌跡間の相関を計算"""
    # 複素相関係数
    z1 = np.array(traj1)
    z2 = np.array(traj2)
    
    # 中心化
    z1_centered = z1 - np.mean(z1)
    z2_centered = z2 - np.mean(z2)
    
    # 複素相関
    correlation = np.sum(z1_centered * np.conj(z2_centered)) / \
                 np.sqrt(np.sum(np.abs(z1_centered)**2) * np.sum(np.abs(z2_centered)**2))
    
    return np.abs(correlation)

def detect_quantum_entanglement(traj1, traj2, threshold=0.7):
    """2つの軌跡間の量子もつれ検出"""
    correlation = compute_complex_correlation(traj1, traj2)
    
    # 位相相関の計算
    phase1 = np.angle(np.array(traj1))
    phase2 = np.angle(np.array(traj2))
    phase_correlation = np.corrcoef(phase1, phase2)[0, 1]
    
    # 振幅相関の計算
    amp1 = np.abs(np.array(traj1))
    amp2 = np.abs(np.array(traj2))
    amplitude_correlation = np.corrcoef(amp1, amp2)[0, 1]
    
    entanglement_score = (correlation + abs(phase_correlation) + abs(amplitude_correlation)) / 3
    
    return {
        'entangled': entanglement_score > threshold,
        'score': entanglement_score,
        'complex_correlation': correlation,
        'phase_correlation': phase_correlation,
        'amplitude_correlation': amplitude_correlation
    }