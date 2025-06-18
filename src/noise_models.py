"""
Quantum Noise Models for CQT Validation
量子ノイズシミュレーション実装

このモジュールは現実的な量子ノイズをシミュレートし、
CQTトラッカーのエラー検出性能を検証するために使用されます。
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum
import matplotlib.pyplot as plt
from dataclasses import dataclass

class NoiseType(Enum):
    """量子ノイズの種類"""
    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping" 
    PHASE_DAMPING = "phase_damping"
    BIT_FLIP = "bit_flip"
    PHASE_FLIP = "phase_flip"

@dataclass
class NoiseParameters:
    """ノイズパラメータ"""
    noise_type: NoiseType
    strength: float  # ノイズ強度 [0, 1]
    correlation_time: int = 10  # 相関時間（測定単位）
    
class QuantumNoiseSimulator:
    """量子ノイズシミュレーター"""
    
    def __init__(self, noise_params: NoiseParameters):
        self.noise_params = noise_params
        self.measurement_count = 0
        self.noise_history = []
        
    def apply_noise_to_state(self, state_vector: np.ndarray) -> np.ndarray:
        """
        量子状態にノイズを適用
        
        Args:
            state_vector: 入力量子状態 [α, β]
            
        Returns:
            ノイズが適用された状態
        """
        if self.noise_params.noise_type == NoiseType.DEPOLARIZING:
            return self._apply_depolarizing_noise(state_vector)
        elif self.noise_params.noise_type == NoiseType.AMPLITUDE_DAMPING:
            return self._apply_amplitude_damping(state_vector)
        elif self.noise_params.noise_type == NoiseType.PHASE_DAMPING:
            return self._apply_phase_damping(state_vector)
        elif self.noise_params.noise_type == NoiseType.BIT_FLIP:
            return self._apply_bit_flip_noise(state_vector)
        elif self.noise_params.noise_type == NoiseType.PHASE_FLIP:
            return self._apply_phase_flip_noise(state_vector)
        else:
            return state_vector
            
    def _apply_depolarizing_noise(self, state: np.ndarray) -> np.ndarray:
        """
        脱分極ノイズ: ρ → (1-p)ρ + p*I/2
        """
        p = self.noise_params.strength
        
        # 確率pで最大混合状態に置き換え
        if np.random.random() < p:
            # 最大混合状態の固有ベクトルをランダム選択
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            noisy_state = np.array([
                np.cos(phi/2),
                np.sin(phi/2) * np.exp(1j * theta)
            ])
            return noisy_state / np.linalg.norm(noisy_state)
        else:
            return state
            
    def _apply_amplitude_damping(self, state: np.ndarray) -> np.ndarray:
        """
        振幅減衰ノイズ: エネルギー散逸をモデル化
        """
        gamma = self.noise_params.strength
        
        # Kraus演算子
        K0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]])
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        
        # 確率的適用
        prob_K1 = gamma * np.abs(state[1])**2
        
        if np.random.random() < prob_K1:
            # K1を適用（|1⟩ → |0⟩の遷移）
            new_state = K1 @ state
        else:
            # K0を適用
            new_state = K0 @ state
            
        # 正規化
        norm = np.linalg.norm(new_state)
        return new_state / norm if norm > 1e-10 else state
        
    def _apply_phase_damping(self, state: np.ndarray) -> np.ndarray:
        """
        位相減衰ノイズ: コヒーレンスの損失
        """
        gamma = self.noise_params.strength
        
        # 位相情報の確率的破壊
        if np.random.random() < gamma:
            # 位相をランダム化
            phase_noise = np.random.uniform(0, 2*np.pi)
            new_state = np.array([
                state[0],
                state[1] * np.exp(1j * phase_noise)
            ])
            return new_state / np.linalg.norm(new_state)
        else:
            return state
            
    def _apply_bit_flip_noise(self, state: np.ndarray) -> np.ndarray:
        """
        ビット反転ノイズ: X演算子の確率的適用
        """
        p = self.noise_params.strength
        
        if np.random.random() < p:
            # Xゲート適用: |0⟩ ↔ |1⟩
            X = np.array([[0, 1], [1, 0]])
            return X @ state
        else:
            return state
            
    def _apply_phase_flip_noise(self, state: np.ndarray) -> np.ndarray:
        """
        位相反転ノイズ: Z演算子の確率的適用
        """
        p = self.noise_params.strength
        
        if np.random.random() < p:
            # Zゲート適用: |1⟩ → -|1⟩
            Z = np.array([[1, 0], [0, -1]])
            return Z @ state
        else:
            return state
            
    def generate_measurement_outcomes(self, clean_state: np.ndarray, 
                                    n_measurements: int) -> List[Tuple[int, np.ndarray]]:
        """
        ノイズのある測定結果を生成
        
        Args:
            clean_state: ノイズなしの理想状態
            n_measurements: 測定回数
            
        Returns:
            (測定結果, ノイズ状態)のリスト
        """
        results = []
        
        for i in range(n_measurements):
            # ノイズ適用
            noisy_state = self.apply_noise_to_state(clean_state.copy())
            
            # 測定確率計算
            prob_0 = np.abs(noisy_state[0])**2
            prob_1 = np.abs(noisy_state[1])**2
            
            # 測定実行
            outcome = 1 if np.random.random() < prob_1 else 0
            
            results.append((outcome, noisy_state))
            self.measurement_count += 1
            
        return results

class NoiseAnalyzer:
    """ノイズ解析クラス"""
    
    @staticmethod
    def compare_clean_vs_noisy_trajectories(clean_trajectory: List[complex],
                                          noisy_trajectory: List[complex]) -> Dict:
        """
        クリーンな軌跡とノイズありの軌跡を比較
        """
        clean_array = np.array(clean_trajectory)
        noisy_array = np.array(noisy_trajectory)
        
        # 統計的指標
        clean_mean = np.mean(clean_array)
        noisy_mean = np.mean(noisy_array)
        
        clean_std = np.std(clean_array)
        noisy_std = np.std(noisy_array)
        
        # 軌跡の違い
        trajectory_distance = np.mean(np.abs(clean_array - noisy_array))
        
        # 位相コヒーレンスの変化
        clean_phases = np.angle(clean_array)
        noisy_phases = np.angle(noisy_array)
        
        clean_coherence = np.abs(np.mean(np.exp(1j * clean_phases)))
        noisy_coherence = np.abs(np.mean(np.exp(1j * noisy_phases)))
        
        return {
            "mean_shift": np.abs(clean_mean - noisy_mean),
            "std_change": noisy_std - clean_std,
            "trajectory_distance": trajectory_distance,
            "coherence_loss": clean_coherence - noisy_coherence,
            "noise_detectability": trajectory_distance / (clean_std + 1e-10)
        }
        
    @staticmethod
    def analyze_noise_signatures(trajectory: List[complex], 
                               window_size: int = 20) -> Dict:
        """
        軌跡からノイズの特徴を抽出
        """
        trajectory_array = np.array(trajectory)
        
        # 移動平均からの偏差
        if len(trajectory) < window_size:
            return {"error": "Insufficient data"}
            
        moving_avg = np.convolve(trajectory_array, 
                                np.ones(window_size)/window_size, 
                                mode='valid')
        
        # 偏差の統計
        deviations = []
        for i in range(len(moving_avg)):
            actual = trajectory_array[i + window_size//2]
            expected = moving_avg[i]
            deviations.append(np.abs(actual - expected))
            
        deviation_array = np.array(deviations)
        
        # フーリエ解析によるノイズ周波数特性
        fft = np.fft.fft(trajectory_array)
        frequencies = np.fft.fftfreq(len(trajectory_array))
        power_spectrum = np.abs(fft)**2
        
        # 高周波ノイズの検出
        high_freq_power = np.sum(power_spectrum[len(power_spectrum)//4:])
        total_power = np.sum(power_spectrum)
        noise_ratio = high_freq_power / total_power
        
        return {
            "deviation_mean": np.mean(deviation_array),
            "deviation_std": np.std(deviation_array),
            "max_deviation": np.max(deviation_array),
            "high_freq_noise_ratio": noise_ratio,
            "suspected_noise_type": NoiseAnalyzer._classify_noise_type(deviation_array, noise_ratio)
        }
        
    @staticmethod
    def _classify_noise_type(deviations: np.ndarray, noise_ratio: float) -> str:
        """
        偏差パターンからノイズ種類を推定
        """
        if noise_ratio > 0.3:
            return "high_frequency_noise_detected"
        elif np.std(deviations) > np.mean(deviations) * 2:
            return "burst_noise_detected"
        elif np.mean(deviations) > 0.1:
            return "systematic_drift_detected"
        else:
            return "low_noise_environment"

def create_noise_test_suite() -> List[NoiseParameters]:
    """
    標準的なノイズテストスイートを作成
    """
    test_suite = []
    
    # 各ノイズタイプで異なる強度をテスト
    noise_types = [NoiseType.DEPOLARIZING, NoiseType.AMPLITUDE_DAMPING, 
                   NoiseType.PHASE_DAMPING, NoiseType.BIT_FLIP, NoiseType.PHASE_FLIP]
    
    strengths = [0.01, 0.05, 0.1, 0.2]  # 1%, 5%, 10%, 20%
    
    for noise_type in noise_types:
        for strength in strengths:
            test_suite.append(NoiseParameters(noise_type, strength))
            
    return test_suite