"""
CQT (Complex Quantum Trajectory) Measurement Tracker - Version 2
物理的制約を厳密に守る改良版

主な改善点：
1. 実部を[-1, +1]に厳密に制限
2. 人工的な時間変調の除去
3. 状態依存の不確実性計算
4. Bell状態の正確なモデル化
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


@dataclass
class MeasurementRecord:
    """測定記録"""
    measurement_index: int
    outcome: int
    complex_value: complex
    timestamp: float = 0.0
    metadata: Dict = field(default_factory=dict)


class ImprovedCQTTracker:
    """改良版CQTトラッカー"""
    
    def __init__(self, system_dim: int = 2):
        self.system_dim = system_dim
        self.measurements: List[MeasurementRecord] = []
        self.trajectory: List[complex] = []
        self.measurement_count = 0
        
    def add_measurement(self, outcome: int, state_vector: Optional[np.ndarray] = None) -> complex:
        """
        測定を追加し、物理的に妥当な複素数表現を計算
        """
        # 方向性（実部）の計算
        direction = self._compute_direction_v2(outcome, state_vector)
        
        # 不確実性（虚部）の計算
        uncertainty = self._compute_uncertainty_v2(outcome, state_vector)
        
        # 複素数の作成
        z = complex(direction, uncertainty)
        
        # 記録の保存
        record = MeasurementRecord(
            measurement_index=self.measurement_count,
            outcome=outcome,
            complex_value=z,
            timestamp=self.measurement_count
        )
        
        self.measurements.append(record)
        self.trajectory.append(z)
        self.measurement_count += 1
        
        return z
    
    def _compute_direction_v2(self, outcome: int, state_vector: Optional[np.ndarray] = None) -> float:
        """
        改良版：物理的に妥当な方向性計算
        """
        if state_vector is not None:
            # 状態ベクトルから期待値を計算
            prob_0 = np.abs(state_vector[0])**2
            prob_1 = np.abs(state_vector[1])**2
            
            # 期待値ベースの方向性
            expectation = prob_1 - prob_0  # ∈ [-1, 1]
            
            # 測定結果による変動（小さなノイズ）
            measurement_bias = 0.1 * (2 * outcome - 1)
            
            # 最近の測定履歴の影響（減衰付き）
            if len(self.trajectory) >= 10:
                recent = self.trajectory[-10:]
                history_influence = 0.1 * np.mean([z.real for z in recent])
            else:
                history_influence = 0.0
            
            # 合成（物理的制約を保証）
            direction = expectation + 0.5 * measurement_bias + 0.3 * history_influence
            
        else:
            # 状態ベクトルがない場合は測定統計から推定
            if len(self.measurements) >= 20:
                recent_outcomes = [m.outcome for m in self.measurements[-20:]]
                p1_estimate = sum(recent_outcomes) / len(recent_outcomes)
                direction = 2 * p1_estimate - 1
            else:
                # デフォルトは測定結果に基づく
                direction = 2 * outcome - 1
        
        # 物理的制約を厳密に適用
        return np.clip(direction, -1.0, 1.0)
    
    def _compute_uncertainty_v2(self, outcome: int, state_vector: Optional[np.ndarray] = None) -> float:
        """
        改良版：状態依存の不確実性計算
        """
        if state_vector is not None:
            # 状態ベクトルから確率を計算
            probabilities = np.abs(state_vector) ** 2
            
            # 純粋な固有状態のチェック
            if np.max(probabilities) > 0.99:  # ほぼ固有状態
                return 0.0  # 不確実性なし
            
            # Shannonエントロピー
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            # 正規化（2準位系の場合、最大エントロピーは1）
            normalized_entropy = entropy / np.log2(self.system_dim)
            
            # 測定結果の履歴も考慮
            if len(self.measurements) >= 10:
                recent_outcomes = [m.outcome for m in self.measurements[-10:]]
                p0 = recent_outcomes.count(0) / len(recent_outcomes)
                p1 = recent_outcomes.count(1) / len(recent_outcomes)
                
                # 履歴エントロピー
                if p0 > 0 and p1 > 0:
                    hist_entropy = -(p0 * np.log2(p0) + p1 * np.log2(p1))
                else:
                    hist_entropy = 0.0
                
                # 状態エントロピーと履歴エントロピーの加重平均
                uncertainty = 0.7 * normalized_entropy + 0.3 * hist_entropy
            else:
                uncertainty = normalized_entropy
                
        else:
            # 状態ベクトルがない場合は測定統計のみ
            if len(self.measurements) >= 10:
                recent_outcomes = [m.outcome for m in self.measurements[-10:]]
                p0 = recent_outcomes.count(0) / len(recent_outcomes)
                p1 = recent_outcomes.count(1) / len(recent_outcomes)
                
                if p0 > 0 and p1 > 0:
                    uncertainty = -(p0 * np.log2(p0) + p1 * np.log2(p1))
                else:
                    uncertainty = 0.0
            else:
                uncertainty = 0.5  # デフォルト
        
        # 物理的制約
        return np.clip(uncertainty, 0.0, 1.0)
    
    def analyze_trajectory(self, window_size: int = 50) -> Dict:
        """軌跡解析（改良版）"""
        if len(self.trajectory) < window_size:
            return {"error": "Not enough measurements for analysis"}
        
        trajectory_array = np.array(self.trajectory)
        
        # 物理的妥当性のチェック
        real_parts = trajectory_array.real
        imag_parts = trajectory_array.imag
        
        validity_checks = {
            "real_part_valid": np.all((-1.0 <= real_parts) & (real_parts <= 1.0)),
            "imag_part_valid": np.all((0.0 <= imag_parts) & (imag_parts <= 1.0)),
            "max_real": np.max(np.abs(real_parts)),
            "max_imag": np.max(imag_parts)
        }
        
        # 標準的な統計
        analysis = {
            "total_measurements": len(self.trajectory),
            "mean_complex": np.mean(trajectory_array),
            "std_complex": np.std(trajectory_array),
            "mean_magnitude": np.mean(np.abs(trajectory_array)),
            "mean_phase": np.mean(np.angle(trajectory_array)),
            "trajectory_length": self._compute_trajectory_length(),
            "phase_coherence": self._compute_phase_coherence(window_size),
            "drift_rate": self._compute_drift_rate(),
            "validity": validity_checks
        }
        
        return analysis
    
    def _compute_trajectory_length(self) -> float:
        """軌跡の総長を計算"""
        if len(self.trajectory) < 2:
            return 0.0
        
        distances = [abs(self.trajectory[i+1] - self.trajectory[i]) 
                    for i in range(len(self.trajectory)-1)]
        return sum(distances)
    
    def _compute_phase_coherence(self, window_size: int) -> float:
        """位相コヒーレンスの計算"""
        if len(self.trajectory) < window_size:
            return 0.0
        
        coherences = []
        for i in range(len(self.trajectory) - window_size + 1):
            window = self.trajectory[i:i+window_size]
            phases = np.angle(window)
            mean_vector = np.mean(np.exp(1j * phases))
            coherence = np.abs(mean_vector)
            coherences.append(coherence)
        
        return np.mean(coherences)
    
    def _compute_drift_rate(self) -> complex:
        """ドリフト率の計算"""
        if len(self.trajectory) < 2:
            return complex(0, 0)
        
        total_drift = self.trajectory[-1] - self.trajectory[0]
        return total_drift / (len(self.trajectory) - 1)


class BellStateTracker:
    """Bell状態用の2量子ビットトラッカー"""
    
    def __init__(self):
        self.qubit1_tracker = ImprovedCQTTracker()
        self.qubit2_tracker = ImprovedCQTTracker()
        self.correlation_history = []
    
    def add_bell_measurement(self, outcome1: int, outcome2: int, 
                           bell_state_type: str) -> Tuple[complex, complex]:
        """
        Bell状態の測定を追加
        """
        # Bell状態に応じた理論的確率
        if bell_state_type == 'phi_plus':  # |00⟩ + |11⟩
            # 完全相関
            state1 = np.array([np.sqrt(0.5), np.sqrt(0.5)])
            state2 = np.array([np.sqrt(0.5), np.sqrt(0.5)])
        elif bell_state_type == 'phi_minus':  # |00⟩ - |11⟩
            # 完全相関（位相差）
            state1 = np.array([np.sqrt(0.5), np.sqrt(0.5)])
            state2 = np.array([np.sqrt(0.5), -np.sqrt(0.5)])
        elif bell_state_type == 'psi_plus':  # |01⟩ + |10⟩
            # 完全反相関
            state1 = np.array([np.sqrt(0.5), np.sqrt(0.5)])
            state2 = np.array([np.sqrt(0.5), np.sqrt(0.5)])
        else:  # psi_minus: |01⟩ - |10⟩
            # 完全反相関（位相差）
            state1 = np.array([np.sqrt(0.5), np.sqrt(0.5)])
            state2 = np.array([np.sqrt(0.5), -np.sqrt(0.5)])
        
        # 各量子ビットの測定を追加
        z1 = self.qubit1_tracker.add_measurement(outcome1, state1)
        z2 = self.qubit2_tracker.add_measurement(outcome2, state2)
        
        # 相関の計算
        correlation = (outcome1 == outcome2) if bell_state_type in ['phi_plus', 'phi_minus'] else (outcome1 != outcome2)
        self.correlation_history.append(correlation)
        
        return z1, z2
    
    def analyze_correlation(self) -> Dict:
        """相関解析"""
        if len(self.correlation_history) < 10:
            return {"error": "Not enough data"}
        
        correlation_strength = np.mean(self.correlation_history)
        
        # 複素軌跡の相関
        traj1 = np.array(self.qubit1_tracker.trajectory)
        traj2 = np.array(self.qubit2_tracker.trajectory)
        
        # 実部と虚部の相関
        real_corr = np.corrcoef(traj1.real, traj2.real)[0, 1]
        imag_corr = np.corrcoef(traj1.imag, traj2.imag)[0, 1]
        
        return {
            "measurement_correlation": correlation_strength,
            "trajectory_real_correlation": real_corr,
            "trajectory_imag_correlation": imag_corr,
            "is_bell_state": correlation_strength > 0.8
        }


def improved_error_detection(trajectory: List[complex], 
                           expected_state_type: str = "unknown") -> str:
    """
    改良版エラー検出
    """
    if len(trajectory) < 20:
        return "INSUFFICIENT_DATA"
    
    recent = trajectory[-20:]
    trajectory_array = np.array(recent)
    
    # 1. デコヒーレンス検出（改良版）
    magnitudes = np.abs(trajectory_array)
    if np.mean(magnitudes[-5:]) < 0.3 and np.std(magnitudes[-5:]) < 0.1:
        return "DECOHERENCE_DETECTED"
    
    # 2. 位相ドリフト検出
    phases = np.angle(trajectory_array)
    phase_diff = np.diff(phases)
    # 循環的な位相差を考慮
    phase_diff = np.mod(phase_diff + np.pi, 2*np.pi) - np.pi
    
    if np.std(phase_diff) > np.pi/2:
        return "PHASE_DRIFT_DETECTED"
    
    # 3. 状態依存のエラー検出
    if expected_state_type == "eigenstate":
        # 固有状態では虚部が小さいはず
        if np.mean(trajectory_array.imag) > 0.2:
            return "STATE_PREPARATION_ERROR"
    elif expected_state_type == "superposition":
        # 重ね合わせ状態では虚部が大きいはず
        if np.mean(trajectory_array.imag) < 0.5:
            return "SUPERPOSITION_COLLAPSE"
    
    # 4. 異常値検出
    real_parts = trajectory_array.real
    imag_parts = trajectory_array.imag
    
    if np.any(np.abs(real_parts) > 1.0) or np.any(imag_parts < 0) or np.any(imag_parts > 1.0):
        return "INVALID_VALUES_DETECTED"
    
    return "NO_ERROR"