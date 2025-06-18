"""
CQT (Complex Quantum Trajectory) Measurement Tracker - Version 3
実験結果に基づいて調整された改良版

主な改善点：
1. 現実的な検出閾値
2. ノイズタイプ別の検出戦略
3. 相対的な変化に基づく検出
4. 状態依存の動的閾値
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


class OptimizedCQTTracker:
    """最適化版CQTトラッカー（v3）"""
    
    def __init__(self, system_dim: int = 2):
        self.system_dim = system_dim
        self.measurements: List[MeasurementRecord] = []
        self.trajectory: List[complex] = []
        self.measurement_count = 0
        # ベースライン統計（最初の50測定から計算）
        self.baseline_stats = None
        
    def add_measurement(self, outcome: int, state_vector: Optional[np.ndarray] = None) -> complex:
        """
        測定を追加し、物理的に妥当な複素数表現を計算
        """
        # 方向性（実部）の計算
        direction = self._compute_direction_v3(outcome, state_vector)
        
        # 不確実性（虚部）の計算
        uncertainty = self._compute_uncertainty_v3(outcome, state_vector)
        
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
        
        # ベースライン統計の計算
        if self.measurement_count == 50:
            self._compute_baseline_stats()
        
        return z
    
    def _compute_direction_v3(self, outcome: int, state_vector: Optional[np.ndarray] = None) -> float:
        """
        v3: より現実的な方向性計算
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
                history_influence = 0.05 * np.mean([z.real for z in recent])
            else:
                history_influence = 0.0
            
            # 合成
            direction = expectation + 0.3 * measurement_bias + 0.2 * history_influence
            
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
    
    def _compute_uncertainty_v3(self, outcome: int, state_vector: Optional[np.ndarray] = None) -> float:
        """
        v3: 現実的な不確実性計算
        """
        if state_vector is not None:
            # 状態ベクトルから確率を計算
            probabilities = np.abs(state_vector) ** 2
            
            # 純粋な固有状態のチェック
            if np.max(probabilities) > 0.98:  # ほぼ固有状態
                return 0.02  # 小さな不確実性
            
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
                uncertainty = 0.8 * normalized_entropy + 0.2 * hist_entropy
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
                    uncertainty = 0.02
            else:
                uncertainty = 0.5  # デフォルト
        
        # 物理的制約（現実的な範囲）
        return np.clip(uncertainty, 0.02, 0.98)
    
    def _compute_baseline_stats(self):
        """ベースライン統計の計算"""
        if len(self.trajectory) >= 50:
            baseline_traj = np.array(self.trajectory[:50])
            self.baseline_stats = {
                'mean_real': np.mean(baseline_traj.real),
                'std_real': np.std(baseline_traj.real),
                'mean_imag': np.mean(baseline_traj.imag),
                'std_imag': np.std(baseline_traj.imag),
                'mean_magnitude': np.mean(np.abs(baseline_traj)),
                'std_magnitude': np.std(np.abs(baseline_traj))
            }
    
    def analyze_trajectory(self, window_size: int = 50) -> Dict:
        """軌跡解析（v3）"""
        if len(self.trajectory) < window_size:
            return {"error": "Not enough measurements for analysis"}
        
        trajectory_array = np.array(self.trajectory)
        
        # 標準的な統計
        analysis = {
            "total_measurements": len(self.trajectory),
            "mean_complex": np.mean(trajectory_array),
            "std_complex": np.std(trajectory_array),
            "mean_magnitude": np.mean(np.abs(trajectory_array)),
            "mean_phase": np.mean(np.angle(trajectory_array)),
            "trajectory_length": self._compute_trajectory_length(),
            "phase_coherence": self._compute_phase_coherence(window_size),
            "drift_rate": self._compute_drift_rate()
        }
        
        # ベースラインとの比較
        if self.baseline_stats:
            recent_traj = trajectory_array[-window_size:]
            analysis['baseline_deviation'] = {
                'real_deviation': abs(np.mean(recent_traj.real) - self.baseline_stats['mean_real']) / (self.baseline_stats['std_real'] + 1e-6),
                'imag_deviation': abs(np.mean(recent_traj.imag) - self.baseline_stats['mean_imag']) / (self.baseline_stats['std_imag'] + 1e-6),
                'magnitude_deviation': abs(np.mean(np.abs(recent_traj)) - self.baseline_stats['mean_magnitude']) / (self.baseline_stats['std_magnitude'] + 1e-6)
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


def optimized_error_detection(trajectory: List[complex], 
                            expected_state_type: str = "unknown",
                            noise_sensitivity: float = 1.0) -> str:
    """
    v3: 最適化されたエラー検出（より積極的な検出）
    
    Args:
        trajectory: 複素軌跡
        expected_state_type: 期待される状態タイプ
        noise_sensitivity: ノイズ感度（1.0が標準、低いほど敏感）
    """
    if len(trajectory) < 30:
        return "INSUFFICIENT_DATA"
    
    # ベースラインと最近の軌跡を分析
    baseline = trajectory[:30]
    recent = trajectory[-30:]
    
    baseline_array = np.array(baseline)
    recent_array = np.array(recent)
    
    # 1. 相対的な変化の検出
    baseline_stats = {
        'mean_real': np.mean(baseline_array.real),
        'std_real': np.std(baseline_array.real),
        'mean_imag': np.mean(baseline_array.imag),
        'std_imag': np.std(baseline_array.imag),
        'mean_magnitude': np.mean(np.abs(baseline_array)),
        'std_magnitude': np.std(np.abs(baseline_array))
    }
    
    recent_stats = {
        'mean_real': np.mean(recent_array.real),
        'std_real': np.std(recent_array.real),
        'mean_imag': np.mean(recent_array.imag),
        'std_imag': np.std(recent_array.imag),
        'mean_magnitude': np.mean(np.abs(recent_array)),
        'std_magnitude': np.std(np.abs(recent_array))
    }
    
    # 2. 異常スコアの計算（より敏感に）
    anomaly_scores = []
    
    # 実部の変化
    real_change = abs(recent_stats['mean_real'] - baseline_stats['mean_real'])
    if baseline_stats['std_real'] > 0.001:
        real_change_normalized = real_change / baseline_stats['std_real']
        anomaly_scores.append(real_change_normalized)
    elif real_change > 0.1:  # 標準偏差が小さい場合は絶対変化を見る
        anomaly_scores.append(real_change * 10)
    
    # 虚部の変化
    imag_change = abs(recent_stats['mean_imag'] - baseline_stats['mean_imag'])
    if baseline_stats['std_imag'] > 0.001:
        imag_change_normalized = imag_change / baseline_stats['std_imag']
        anomaly_scores.append(imag_change_normalized)
    elif imag_change > 0.05:  # 小さな変化でも検出
        anomaly_scores.append(imag_change * 20)
    
    # 絶対値の変化
    magnitude_change = abs(recent_stats['mean_magnitude'] - baseline_stats['mean_magnitude'])
    if baseline_stats['std_magnitude'] > 0.001:
        magnitude_change_normalized = magnitude_change / baseline_stats['std_magnitude']
        anomaly_scores.append(magnitude_change_normalized)
    
    # 標準偏差の増加（ノイズの重要指標）
    std_increase_real = recent_stats['std_real'] / (baseline_stats['std_real'] + 0.001)
    std_increase_imag = recent_stats['std_imag'] / (baseline_stats['std_imag'] + 0.001)
    
    if std_increase_real > 1.5:  # 1.5倍以上の増加
        anomaly_scores.append((std_increase_real - 1) * 2)
    if std_increase_imag > 1.5:
        anomaly_scores.append((std_increase_imag - 1) * 2)
    
    # 3. 状態固有の検出（より敏感に）
    if expected_state_type == "eigenstate":
        # 固有状態では虚部が非常に小さいはず
        if recent_stats['mean_imag'] > 0.1:  # 閾値を下げる
            return "STATE_PREPARATION_ERROR"
        # 固有状態では標準偏差も小さいはず
        if recent_stats['std_real'] > 0.15 or recent_stats['std_imag'] > 0.15:
            return "EIGENSTATE_NOISE_DETECTED"
    elif expected_state_type == "superposition":
        # 重ね合わせ状態では虚部がある程度大きいはず
        if recent_stats['mean_imag'] < 0.4:  # より厳しく
            # 虚部の減少をチェック
            if baseline_stats['mean_imag'] > 0.6:  # ベースラインが高い場合
                imag_decrease = (baseline_stats['mean_imag'] - recent_stats['mean_imag']) / baseline_stats['mean_imag']
                if imag_decrease > 0.2:  # 20%以上の減少
                    return "SUPERPOSITION_COLLAPSE"
    
    # 4. 総合的な異常判定（より積極的に）
    if anomaly_scores:
        max_anomaly = max(anomaly_scores)
        mean_anomaly = np.mean(anomaly_scores)
        
        # より低い閾値で検出
        threshold = 1.5 / noise_sensitivity  # 1.5標準偏差から検出
        
        if max_anomaly > threshold or mean_anomaly > threshold * 0.7:
            # 異常の種類を特定
            if std_increase_real > 2.0 or std_increase_imag > 2.0:
                return "NOISE_INCREASE_DETECTED"
            elif magnitude_change > 0.1 and recent_stats['mean_magnitude'] < baseline_stats['mean_magnitude']:
                return "DECOHERENCE_DETECTED"
            elif real_change > 0.2:
                return "PHASE_DRIFT_DETECTED"
            elif imag_change > 0.15:
                return "UNCERTAINTY_CHANGE_DETECTED"
            else:
                return "ANOMALY_DETECTED"
    
    # 5. 追加の検出方法：軌跡のジグザグ度
    if len(trajectory) > 50:
        # 最近の軌跡の変化率を計算
        recent_diffs = np.diff(recent_array)
        zigzag_score = np.mean(np.abs(recent_diffs))
        
        # ベースラインの変化率
        baseline_diffs = np.diff(baseline_array)
        baseline_zigzag = np.mean(np.abs(baseline_diffs))
        
        if zigzag_score > baseline_zigzag * 2.0:  # 2倍以上のジグザグ
            return "TRAJECTORY_INSTABILITY_DETECTED"
    
    # 6. 物理的制約チェック
    real_parts = recent_array.real
    imag_parts = recent_array.imag
    
    if np.any(np.abs(real_parts) > 1.0) or np.any(imag_parts < 0) or np.any(imag_parts > 1.0):
        return "INVALID_VALUES_DETECTED"
    
    return "NO_ERROR"


# 既存のクラスとの互換性のためのエイリアス
ImprovedCQTTracker = OptimizedCQTTracker
improved_error_detection = optimized_error_detection