"""
CQT (Complex Quantum Trajectory) Measurement Tracker

This module implements the core CQT theory for quantum measurement analysis.
The key idea is to represent each measurement as a complex number:
z = (direction) + i*(uncertainty)

Author: CQT Research Team
Date: 2024
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


def simulate_measurements(state_type: str, n_measurements: int) -> List[int]:
    """
    Simulate quantum measurements for different state types
    
    Args:
        state_type: Type of quantum state ('eigenstate_0', 'eigenstate_1', 
                   'superposition_plus', 'superposition_minus', etc.)
        n_measurements: Number of measurements to simulate
    
    Returns:
        List of measurement outcomes (0 or 1)
    """
    if state_type == 'eigenstate_0':
        # |0⟩ state: always measure 0
        return [0] * n_measurements
    elif state_type == 'eigenstate_1':
        # |1⟩ state: always measure 1
        return [1] * n_measurements
    elif state_type == 'superposition_plus':
        # |+⟩ = (|0⟩ + |1⟩)/√2: 50/50 probability
        return np.random.randint(0, 2, n_measurements).tolist()
    elif state_type == 'superposition_minus':
        # |-⟩ = (|0⟩ - |1⟩)/√2: 50/50 probability
        return np.random.randint(0, 2, n_measurements).tolist()
    elif state_type == 'superposition_i':
        # |i⟩ = (|0⟩ + i|1⟩)/√2: 50/50 probability
        return np.random.randint(0, 2, n_measurements).tolist()
    elif state_type == 'superposition_minus_i':
        # |-i⟩ = (|0⟩ - i|1⟩)/√2: 50/50 probability
        return np.random.randint(0, 2, n_measurements).tolist()
    else:
        # Default: equal superposition
        return np.random.randint(0, 2, n_measurements).tolist()


@dataclass
class MeasurementRecord:
    """Single measurement record with complex representation"""
    measurement_index: int
    outcome: int  # 0 or 1 for qubit
    complex_value: complex
    timestamp: float = 0.0
    metadata: Dict = field(default_factory=dict)


class CQTMeasurementTracker:
    """
    Complex Quantum Trajectory (CQT) Measurement Tracker
    
    This class tracks quantum measurements using complex number representation
    where each measurement is encoded as z = direction + i*uncertainty.
    """
    
    def __init__(self, system_dim: int = 2):
        """
        Initialize CQT tracker
        
        Args:
            system_dim: Dimension of quantum system (2 for qubit, 3 for qutrit, etc.)
        """
        self.system_dim = system_dim
        self.measurements: List[MeasurementRecord] = []
        self.trajectory: List[complex] = []
        self.measurement_count = 0
        
    def add_measurement(self, outcome: int, state_vector: Optional[np.ndarray] = None) -> complex:
        """
        Add a quantum measurement and compute its complex representation
        
        Args:
            outcome: Measurement outcome (0 or 1 for qubit)
            state_vector: Optional quantum state vector at measurement time
            
        Returns:
            Complex number representing the measurement
        """
        # Compute direction component (real part)
        direction = self._compute_direction(outcome, self.measurement_count)
        
        # Compute uncertainty component (imaginary part)
        uncertainty = self._compute_uncertainty(outcome, state_vector)
        
        # Create complex representation
        z = complex(direction, uncertainty)
        
        # Store measurement record
        record = MeasurementRecord(
            measurement_index=self.measurement_count,
            outcome=outcome,
            complex_value=z,
            timestamp=self.measurement_count  # Simple timestamp for now
        )
        
        self.measurements.append(record)
        self.trajectory.append(z)
        self.measurement_count += 1
        
        return z
    
    def _compute_direction(self, outcome: int, index: int) -> float:
        """
        Compute direction component based on measurement outcome and history
        
        Args:
            outcome: Current measurement outcome
            index: Measurement index in sequence
            
        Returns:
            Direction value (real component)
        """
        # Base direction from outcome
        base_direction = 2 * outcome - 1  # Maps 0->-1, 1->+1
        
        # Add temporal modulation based on measurement index
        temporal_factor = np.cos(2 * np.pi * index / 100)  # Period of 100 measurements
        
        # Include history effect if we have previous measurements
        history_effect = 0.0
        if len(self.trajectory) > 0:
            # Recent history influence (last 10 measurements)
            recent_history = self.trajectory[-10:]
            history_effect = np.mean([z.real for z in recent_history]) * 0.3
        
        return base_direction + 0.2 * temporal_factor + history_effect
    
    def _compute_uncertainty(self, outcome: int, state_vector: Optional[np.ndarray] = None) -> float:
        """
        Compute uncertainty component based on quantum state
        
        Args:
            outcome: Measurement outcome
            state_vector: Optional quantum state vector
            
        Returns:
            Uncertainty value (imaginary component)
        """
        if state_vector is not None:
            # Use quantum state to compute uncertainty
            probabilities = np.abs(state_vector) ** 2
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            uncertainty = entropy / np.log(self.system_dim)  # Normalized entropy
        else:
            # Use outcome statistics for uncertainty estimate
            if len(self.measurements) > 10:
                recent_outcomes = [m.outcome for m in self.measurements[-10:]]
                p0 = recent_outcomes.count(0) / len(recent_outcomes)
                p1 = recent_outcomes.count(1) / len(recent_outcomes)
                # Binary entropy
                uncertainty = -(p0 * np.log(p0 + 1e-10) + p1 * np.log(p1 + 1e-10)) / np.log(2)
            else:
                # Default uncertainty for early measurements
                uncertainty = 0.5
        
        return uncertainty
    
    def analyze_trajectory(self, window_size: int = 50) -> Dict:
        """
        Analyze the measurement trajectory for patterns
        
        Args:
            window_size: Size of sliding window for analysis
            
        Returns:
            Dictionary containing analysis results
        """
        if len(self.trajectory) < window_size:
            return {"error": "Not enough measurements for analysis"}
        
        trajectory_array = np.array(self.trajectory)
        
        # Compute trajectory statistics
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
        
        return analysis
    
    def _compute_trajectory_length(self) -> float:
        """Compute the total length of the trajectory in complex plane"""
        if len(self.trajectory) < 2:
            return 0.0
        
        distances = [abs(self.trajectory[i+1] - self.trajectory[i]) 
                    for i in range(len(self.trajectory)-1)]
        return sum(distances)
    
    def _compute_phase_coherence(self, window_size: int) -> float:
        """
        Compute phase coherence measure over sliding windows
        
        Args:
            window_size: Size of analysis window
            
        Returns:
            Phase coherence value between 0 and 1
        """
        if len(self.trajectory) < window_size:
            return 0.0
        
        coherences = []
        for i in range(len(self.trajectory) - window_size + 1):
            window = self.trajectory[i:i+window_size]
            phases = np.angle(window)
            # Compute circular variance
            mean_vector = np.mean(np.exp(1j * phases))
            coherence = np.abs(mean_vector)
            coherences.append(coherence)
        
        return np.mean(coherences)
    
    def _compute_drift_rate(self) -> complex:
        """Compute the average drift rate in complex plane"""
        if len(self.trajectory) < 2:
            return complex(0, 0)
        
        total_drift = self.trajectory[-1] - self.trajectory[0]
        return total_drift / (len(self.trajectory) - 1)
    
    def plot_trajectory(self, title: str = "CQT Measurement Trajectory", 
                       figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot the measurement trajectory in complex plane
        
        Args:
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Complex plane trajectory
        trajectory_array = np.array(self.trajectory)
        real_parts = trajectory_array.real
        imag_parts = trajectory_array.imag
        
        # Color map for temporal evolution
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.trajectory)))
        
        # Plot trajectory with color gradient
        for i in range(len(self.trajectory) - 1):
            ax1.plot(real_parts[i:i+2], imag_parts[i:i+2], 
                    color=colors[i], alpha=0.7, linewidth=2)
        
        # Mark start and end points
        ax1.scatter(real_parts[0], imag_parts[0], color='green', 
                   s=100, label='Start', zorder=5)
        ax1.scatter(real_parts[-1], imag_parts[-1], color='red', 
                   s=100, label='End', zorder=5)
        
        ax1.set_xlabel('Direction (Real)')
        ax1.set_ylabel('Uncertainty (Imaginary)')
        ax1.set_title(f'{title} - Complex Plane')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')
        
        # Time series of magnitude and phase
        magnitudes = np.abs(trajectory_array)
        phases = np.angle(trajectory_array)
        
        ax2.plot(magnitudes, label='Magnitude', color='blue', alpha=0.7)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(phases, label='Phase', color='orange', alpha=0.7)
        
        ax2.set_xlabel('Measurement Index')
        ax2.set_ylabel('Magnitude', color='blue')
        ax2_twin.set_ylabel('Phase (radians)', color='orange')
        ax2.set_title('Magnitude and Phase Evolution')
        ax2.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def detect_bell_signature(self, correlation_threshold: float = 0.8) -> bool:
        """
        Detect potential Bell state signature in measurement trajectory
        
        Args:
            correlation_threshold: Threshold for correlation detection
            
        Returns:
            Boolean indicating if Bell-like correlations detected
        """
        if len(self.trajectory) < 100:
            return False
        
        # Analyze trajectory for characteristic Bell state patterns
        # This is a simplified version - real implementation would be more sophisticated
        
        # Check for anti-correlation patterns in real/imaginary parts
        real_parts = np.array([z.real for z in self.trajectory])
        imag_parts = np.array([z.imag for z in self.trajectory])
        
        # Compute autocorrelation
        real_autocorr = np.correlate(real_parts, real_parts, mode='same')
        imag_autocorr = np.correlate(imag_parts, imag_parts, mode='same')
        
        # Normalize
        real_autocorr = real_autocorr / np.max(real_autocorr)
        imag_autocorr = imag_autocorr / np.max(imag_autocorr)
        
        # Check for characteristic oscillation pattern
        mid_point = len(real_autocorr) // 2
        correlation_strength = np.mean(np.abs(real_autocorr[mid_point-10:mid_point+10]))
        
        return correlation_strength > correlation_threshold
    
    def export_data(self) -> Dict:
        """Export measurement data for further analysis"""
        return {
            "measurements": [(m.measurement_index, m.outcome, m.complex_value) 
                           for m in self.measurements],
            "trajectory": self.trajectory,
            "analysis": self.analyze_trajectory(),
            "system_dim": self.system_dim,
            "measurement_count": self.measurement_count
        }