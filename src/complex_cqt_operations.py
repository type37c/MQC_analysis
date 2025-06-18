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
            # ゼロ除算を防ぐ
            vel_abs = np.abs(velocity[:-1])
            non_zero_mask = vel_abs > 1e-10
            curvature = np.zeros(len(velocity) - 1)
            if np.any(non_zero_mask):
                curvature[non_zero_mask] = np.imag(acceleration[non_zero_mask] * np.conj(velocity[:-1][non_zero_mask])) / (vel_abs[non_zero_mask]**3)
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
            if lag >= n or lag == 0:
                return 1.0 if lag == 0 else 0.0
            if n - lag <= 0:
                return 0.0
            
            z_mean_sq = np.mean(np.abs(z)**2)
            if z_mean_sq == 0:
                return 0.0
            
            return np.mean(z[:-lag] * np.conj(z[lag:])) / z_mean_sq
        
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