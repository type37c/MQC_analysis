# 🚀 複素演算を活用したCQT理論の深化プロジェクト

## 📋 実装する複素演算解析ツール

### 1. complex_cqt_operations.py
```python
"""
CQT理論のための複素演算ライブラリ
W字軌跡の数学的特性を解明
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
import seaborn as sns

class ComplexCQTAnalyzer:
    """複素演算によるCQT軌跡の高度な解析"""
    
    def __init__(self, trajectory_data):
        """
        trajectory_data: 複素数の配列 (numpy array of complex numbers)
        """
        self.trajectory = np.array(trajectory_data)
        self.length = len(trajectory_data)
        
    def compute_instantaneous_properties(self):
        """瞬間的な物理量を計算"""
        # 複素速度
        dt = 1  # 測定間隔
        velocity = np.diff(self.trajectory) / dt
        
        # 複素加速度
        acceleration = np.diff(velocity) / dt
        
        # 瞬間周波数（位相の時間微分）
        phase = np.unwrap(np.angle(self.trajectory))
        instantaneous_frequency = np.diff(phase) / dt
        
        # 瞬間振幅変化率
        amplitude = np.abs(self.trajectory)
        amplitude_rate = np.diff(amplitude) / dt
        
        # 曲率（複素数ならではの計算）
        if len(velocity) > 1:
            curvature = np.imag(acceleration * np.conj(velocity[:-1])) / (np.abs(velocity[:-1])**3)
        else:
            curvature = np.array([])
        
        return {
            'velocity': velocity,
            'acceleration': acceleration,
            'instant_frequency': instantaneous_frequency,
            'amplitude_rate': amplitude_rate,
            'curvature': curvature,
            'speed': np.abs(velocity),
            'direction': np.angle(velocity)
        }
    
    def analyze_w_pattern(self):
        """W字パターンの複素数的特徴を抽出"""
        # 1. 巻き数（Winding Number）
        angles = np.angle(self.trajectory)
        unwrapped = np.unwrap(angles)
        total_rotation = unwrapped[-1] - unwrapped[0]
        winding_number = total_rotation / (2 * np.pi)
        
        # 2. 複素モーメント（形状の特徴）
        centroid = np.mean(self.trajectory)
        centered = self.trajectory - centroid
        
        moments = {
            'order_1': np.mean(centered),
            'order_2': np.mean(centered**2),
            'order_3': np.mean(centered**3),
            'order_4': np.mean(centered**4)
        }
        
        # 3. 自己相関関数（複素版）
        def complex_autocorrelation(z, lag):
            n = len(z)
            if lag >= n:
                return 0
            return np.mean(z[:-lag] * np.conj(z[lag:])) / np.mean(np.abs(z)**2)
        
        max_lag = min(50, self.length // 2)
        autocorr = [complex_autocorrelation(self.trajectory, lag) 
                   for lag in range(max_lag)]
        
        # 4. フラクタル次元の推定
        def box_counting_dimension(z, epsilon_range):
            dimensions = []
            for epsilon in epsilon_range:
                # 複素平面をグリッドに分割
                real_bins = np.arange(z.real.min(), z.real.max() + epsilon, epsilon)
                imag_bins = np.arange(z.imag.min(), z.imag.max() + epsilon, epsilon)
                
                # 占有されているボックスを数える
                hist, _, _ = np.histogram2d(z.real, z.imag, bins=[real_bins, imag_bins])
                n_boxes = np.sum(hist > 0)
                
                if n_boxes > 0:
                    dimensions.append((np.log(1/epsilon), np.log(n_boxes)))
            
            if len(dimensions) > 1:
                # 線形フィッティングで次元を推定
                x, y = zip(*dimensions)
                slope, _ = np.polyfit(x, y, 1)
                return slope
            return None
        
        epsilon_range = np.logspace(-2, -0.5, 10)
        fractal_dim = box_counting_dimension(self.trajectory, epsilon_range)
        
        return {
            'winding_number': winding_number,
            'moments': moments,
            'autocorrelation': autocorr,
            'fractal_dimension': fractal_dim,
            'centroid': centroid,
            'spread': np.std(np.abs(centered))
        }
    
    def fourier_analysis(self):
        """複素フーリエ解析"""
        # 複素数列のフーリエ変換
        fft_result = np.fft.fft(self.trajectory)
        frequencies = np.fft.fftfreq(self.length)
        
        # パワースペクトル
        power_spectrum = np.abs(fft_result)**2
        
        # 主要周波数成分
        idx_sorted = np.argsort(power_spectrum)[::-1]
        dominant_freqs = frequencies[idx_sorted[:5]]
        dominant_powers = power_spectrum[idx_sorted[:5]]
        
        # スペクトルエントロピー（周波数分布の複雑さ）
        normalized_power = power_spectrum / np.sum(power_spectrum)
        spectral_entropy = -np.sum(normalized_power * np.log(normalized_power + 1e-10))
        
        # 位相スペクトル
        phase_spectrum = np.angle(fft_result)
        
        return {
            'fft': fft_result,
            'frequencies': frequencies,
            'power_spectrum': power_spectrum,
            'phase_spectrum': phase_spectrum,
            'dominant_frequencies': dominant_freqs,
            'dominant_powers': dominant_powers,
            'spectral_entropy': spectral_entropy
        }
    
    def detect_phase_transitions(self):
        """軌跡中の相転移点を検出"""
        properties = self.compute_instantaneous_properties()
        
        # 1. 速度の急激な変化点
        speed = properties['speed']
        speed_changes = np.abs(np.diff(speed))
        threshold = np.mean(speed_changes) + 2 * np.std(speed_changes)
        velocity_transitions = np.where(speed_changes > threshold)[0]
        
        # 2. 方向の急激な変化点
        direction = properties['direction']
        direction_changes = np.abs(np.diff(direction))
        # 角度の巻き戻しを考慮
        direction_changes = np.minimum(direction_changes, 2*np.pi - direction_changes)
        dir_threshold = np.mean(direction_changes) + 2 * np.std(direction_changes)
        direction_transitions = np.where(direction_changes > dir_threshold)[0]
        
        # 3. 曲率の極値点
        if len(properties['curvature']) > 0:
            curvature_peaks = signal.find_peaks(np.abs(properties['curvature']))[0]
        else:
            curvature_peaks = np.array([])
        
        return {
            'velocity_transitions': velocity_transitions,
            'direction_transitions': direction_transitions,
            'curvature_peaks': curvature_peaks,
            'all_transitions': np.unique(np.concatenate([
                velocity_transitions,
                direction_transitions,
                curvature_peaks
            ]))
        }
    
    def quantum_state_reconstruction(self):
        """複素軌跡から量子状態を推定"""
        # 軌跡を量子状態として解釈
        # z = <0|ψ> + i<1|ψ> と仮定
        
        # 正規化
        norms = np.abs(self.trajectory)
        max_norm = np.max(norms)
        
        # 各点での量子状態を推定
        quantum_states = []
        for z in self.trajectory:
            # 複素振幅から確率振幅を計算
            alpha = z.real / max_norm
            beta = z.imag / max_norm
            
            # 正規化
            norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
            if norm > 0:
                alpha /= norm
                beta /= norm
            
            # ブロッホ球表現
            theta = 2 * np.arccos(abs(alpha))
            phi = np.angle(beta) - np.angle(alpha)
            
            quantum_states.append({
                'alpha': alpha,
                'beta': beta,
                'theta': theta,
                'phi': phi,
                'prob_0': abs(alpha)**2,
                'prob_1': abs(beta)**2
            })
        
        return pd.DataFrame(quantum_states)
    
    def calculate_geometric_invariants(self):
        """幾何学的不変量の計算"""
        # 1. 全長（複素積分）
        velocity = np.diff(self.trajectory)
        total_length = np.sum(np.abs(velocity))
        
        # 2. 囲む面積（グリーンの定理）
        z = self.trajectory
        area = 0.5 * np.abs(np.sum(z[:-1] * np.conj(z[1:]) - z[1:] * np.conj(z[:-1])))
        
        # 3. 重心からの平均距離
        centroid = np.mean(z)
        mean_distance = np.mean(np.abs(z - centroid))
        
        # 4. 回転不変モーメント
        centered = z - centroid
        I_0 = np.mean(np.abs(centered)**2)  # 慣性モーメント
        I_2 = np.abs(np.mean(centered**2))  # 2次モーメント
        
        # 5. 形状の非対称性
        asymmetry = I_2 / I_0 if I_0 > 0 else 0
        
        return {
            'total_length': total_length,
            'enclosed_area': area,
            'mean_distance': mean_distance,
            'moment_of_inertia': I_0,
            'asymmetry': asymmetry,
            'compactness': area / (total_length**2) if total_length > 0 else 0
        }

def visualize_complex_analysis(analyzer, save_prefix='complex_analysis'):
    """複素解析結果の包括的な可視化"""
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 軌跡と瞬間的性質
    ax1 = plt.subplot(3, 3, 1)
    trajectory = analyzer.trajectory
    props = analyzer.compute_instantaneous_properties()
    
    # 速度によって色分け
    points = np.array([trajectory.real, trajectory.imag]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, cmap='viridis')
    lc.set_array(props['speed'])
    ax1.add_collection(lc)
    ax1.autoscale()
    plt.colorbar(lc, ax=ax1, label='Speed')
    ax1.set_title('Trajectory colored by speed')
    ax1.set_xlabel('Real')
    ax1.set_ylabel('Imaginary')
    
    # 2. 瞬間周波数
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(props['instant_frequency'], 'b-', alpha=0.7)
    ax2.set_title('Instantaneous Frequency')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Frequency (rad/step)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 曲率
    ax3 = plt.subplot(3, 3, 3)
    if len(props['curvature']) > 0:
        ax3.plot(props['curvature'], 'r-', alpha=0.7)
        ax3.set_title('Curvature')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Curvature')
        ax3.grid(True, alpha=0.3)
    
    # 4. 複素自己相関
    ax4 = plt.subplot(3, 3, 4)
    w_features = analyzer.analyze_w_pattern()
    autocorr = w_features['autocorrelation']
    ax4.plot(np.abs(autocorr), 'g-', label='Magnitude')
    ax4.plot(np.real(autocorr), 'b--', label='Real', alpha=0.7)
    ax4.plot(np.imag(autocorr), 'r--', label='Imaginary', alpha=0.7)
    ax4.set_title('Complex Autocorrelation')
    ax4.set_xlabel('Lag')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. パワースペクトル
    ax5 = plt.subplot(3, 3, 5)
    fourier = analyzer.fourier_analysis()
    freqs = fourier['frequencies'][:len(fourier['frequencies'])//2]
    power = fourier['power_spectrum'][:len(fourier['power_spectrum'])//2]
    ax5.semilogy(freqs, power, 'purple')
    ax5.set_title('Power Spectrum')
    ax5.set_xlabel('Frequency')
    ax5.set_ylabel('Power')
    ax5.grid(True, alpha=0.3)
    
    # 6. 位相スペクトル
    ax6 = plt.subplot(3, 3, 6)
    phase = fourier['phase_spectrum'][:len(fourier['phase_spectrum'])//2]
    ax6.plot(freqs, phase, 'orange')
    ax6.set_title('Phase Spectrum')
    ax6.set_xlabel('Frequency')
    ax6.set_ylabel('Phase (rad)')
    ax6.grid(True, alpha=0.3)
    
    # 7. 量子状態の進化
    ax7 = plt.subplot(3, 3, 7)
    quantum_states = analyzer.quantum_state_reconstruction()
    ax7.plot(quantum_states['prob_0'], 'b-', label='|0⟩ probability')
    ax7.plot(quantum_states['prob_1'], 'r-', label='|1⟩ probability')
    ax7.set_title('Quantum State Evolution')
    ax7.set_xlabel('Time')
    ax7.set_ylabel('Probability')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. ブロッホ球の軌跡（2D投影）
    ax8 = plt.subplot(3, 3, 8)
    theta = quantum_states['theta']
    phi = quantum_states['phi']
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    ax8.plot(x, y, 'k-', alpha=0.5)
    ax8.scatter(x[0], y[0], color='green', s=100, label='Start')
    ax8.scatter(x[-1], y[-1], color='red', s=100, label='End')
    ax8.set_title('Bloch Sphere Trajectory (2D)')
    ax8.set_xlabel('X')
    ax8.set_ylabel('Y')
    ax8.legend()
    ax8.set_aspect('equal')
    
    # 9. 特徴サマリー
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # 特徴をテキストで表示
    invariants = analyzer.calculate_geometric_invariants()
    transitions = analyzer.detect_phase_transitions()
    
    summary_text = f"""
    W-Pattern Features:
    - Winding Number: {w_features['winding_number']:.3f}
    - Fractal Dimension: {w_features['fractal_dimension']:.3f}
    - Spectral Entropy: {fourier['spectral_entropy']:.3f}
    
    Geometric Invariants:
    - Total Length: {invariants['total_length']:.3f}
    - Enclosed Area: {invariants['enclosed_area']:.3f}
    - Asymmetry: {invariants['asymmetry']:.3f}
    
    Phase Transitions:
    - Velocity: {len(transitions['velocity_transitions'])} points
    - Direction: {len(transitions['direction_transitions'])} points
    - Curvature: {len(transitions['curvature_peaks'])} peaks
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", 
                                              facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('Complex CQT Analysis Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_complete.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# 実行例
def run_complex_analysis(trajectory_data, name='experiment'):
    """
    trajectory_data: 複素数の配列またはリスト
    """
    print(f"=== {name} の複素演算解析 ===")
    
    analyzer = ComplexCQTAnalyzer(trajectory_data)
    
    # 1. 瞬間的性質
    instant_props = analyzer.compute_instantaneous_properties()
    print(f"\n瞬間的性質:")
    print(f"  平均速度: {np.mean(instant_props['speed']):.4f}")
    print(f"  最大加速度: {np.max(np.abs(instant_props['acceleration'])):.4f}")
    
    # 2. W字パターン解析
    w_features = analyzer.analyze_w_pattern()
    print(f"\nW字パターンの特徴:")
    print(f"  巻き数: {w_features['winding_number']:.3f}")
    print(f"  フラクタル次元: {w_features['fractal_dimension']:.3f}")
    
    # 3. フーリエ解析
    fourier = analyzer.fourier_analysis()
    print(f"\nフーリエ解析:")
    print(f"  主要周波数: {fourier['dominant_frequencies'][:3]}")
    print(f"  スペクトルエントロピー: {fourier['spectral_entropy']:.3f}")
    
    # 4. 幾何学的不変量
    invariants = analyzer.calculate_geometric_invariants()
    print(f"\n幾何学的不変量:")
    print(f"  全長: {invariants['total_length']:.3f}")
    print(f"  囲む面積: {invariants['enclosed_area']:.3f}")
    print(f"  コンパクト性: {invariants['compactness']:.4f}")
    
    # 5. 相転移検出
    transitions = analyzer.detect_phase_transitions()
    print(f"\n相転移点:")
    print(f"  検出された転移点総数: {len(transitions['all_transitions'])}")
    
    # 可視化
    visualize_complex_analysis(analyzer, save_prefix=f'complex_{name}')
    
    return analyzer
```

### 2. エラー検出の複素演算版

```python
# complex_error_detection.py
"""
複素演算によるエラー検出の高度化
"""

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
```

### 3. 実行スクリプト

```python
# run_complex_cqt_analysis.py
"""
複素演算を活用したCQT解析の実行
"""
import numpy as np
import pandas as pd
from complex_cqt_operations import ComplexCQTAnalyzer, run_complex_analysis
from complex_error_detection import ComplexErrorDetector

def main():
    print("=== 複素演算によるCQT解析開始 ===")
    
    # 1. データの読み込み（あなたの実データ）
    # ここは実際のデータパスに置き換えてください
    trajectory_data = load_your_w_trajectory()  # 複素数の配列
    
    # 2. 複素演算解析の実行
    analyzer = run_complex_analysis(trajectory_data, name='IBM_W_pattern')
    
    # 3. エラー検出の実行
    print("\n=== 複素演算によるエラー検出 ===")
    
    # 参照軌跡（エラーなし）とテスト軌跡（エラーあり）
    clean_trajectory = load_clean_trajectory()
    noisy_trajectory = load_noisy_trajectory()
    
    detector = ComplexErrorDetector(clean_trajectory)
    errors = detector.detect_errors(noisy_trajectory)
    
    print(f"\n検出されたエラー数: {len(errors)}")
    for error in errors[:5]:  # 最初の5個を表示
        error_type = detector.classify_error_type(error)
        print(f"  位置{error['position']}: {error_type} (深刻度: {error['severity']:.2f})")
    
    # 4. 複素相関解析
    print("\n=== 複素相関解析 ===")
    
    # 2つの量子ビット間の相関
    if hasattr(trajectory_data, 'qubit1') and hasattr(trajectory_data, 'qubit2'):
        correlation = compute_complex_correlation(
            trajectory_data.qubit1, 
            trajectory_data.qubit2
        )
        print(f"複素相関係数: {correlation:.4f}")
    
    print("\n=== 解析完了 ===")

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

if __name__ == "__main__":
    main()
```

## 🎯 実装する複素演算の特徴

1. **瞬間的物理量**
   - 複素速度・加速度
   - 瞬間周波数
   - 曲率

2. **W字パターンの数学的特徴**
   - 巻き数
   - フラクタル次元
   - 複素モーメント

3. **フーリエ解析**
   - パワースペクトル
   - 位相スペクトル
   - 主要周波数成分

4. **幾何学的不変量**
   - 軌跡の全長
   - 囲む面積
   - 形状の非対称性

5. **量子状態推定**
   - ブロッホ球表現
   - 確率振幅の時間発展

これらの複素演算により、単なる2Dプロットでは見えない深い物理的・数学的構造が明らかになります！