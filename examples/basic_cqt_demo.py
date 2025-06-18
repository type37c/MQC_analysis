#!/usr/bin/env python3
"""
Basic CQT (Complex Quantum Trajectory) Analysis Demo

This script demonstrates the basic usage of the CQT analysis framework
using synthetic quantum measurement data.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cqt_tracker_v3 import OptimizedCQTTracker


def generate_bell_state_measurements(n_measurements=1000, error_rate=0.0):
    """Generate synthetic Bell state measurement data."""
    measurements = np.random.choice([0, 1], size=n_measurements, p=[0.5, 0.5])
    
    # Add errors
    if error_rate > 0:
        n_errors = int(n_measurements * error_rate)
        error_indices = np.random.choice(n_measurements, n_errors, replace=False)
        measurements[error_indices] = 1 - measurements[error_indices]
    
    return measurements


def generate_noisy_measurements(n_measurements=1000, noise_level=0.3):
    """Generate noisy quantum measurement data."""
    # Base pattern with some structure
    base_prob = 0.5 + 0.2 * np.sin(np.linspace(0, 4*np.pi, n_measurements))
    
    # Add noise
    noise = np.random.normal(0, noise_level, n_measurements)
    prob = np.clip(base_prob + noise, 0, 1)
    
    measurements = np.random.binomial(1, prob)
    return measurements


def main():
    """Main demonstration function."""
    print("🚀 CQT Analysis Demo Starting...")
    print("=" * 50)
    
    # Generate example data
    print("📊 Generating example data...")
    bell_measurements = generate_bell_state_measurements(1000, error_rate=0.05)
    noisy_measurements = generate_noisy_measurements(1000, noise_level=0.4)
    
    # Analyze Bell state data
    print("\n🔬 Analyzing Bell state data...")
    bell_tracker = OptimizedCQTTracker()
    for measurement in bell_measurements:
        bell_tracker.add_measurement(measurement)
    
    bell_trajectory = bell_tracker.get_trajectory()
    bell_analysis = bell_tracker.analyze_trajectory()
    bell_errors = bell_tracker.detect_errors()
    
    print(f"Bell State Results:")
    print(f"  • Trajectory Length: {len(bell_trajectory)}")
    print(f"  • Mean Magnitude: {np.mean(np.abs(bell_trajectory)):.3f}")
    print(f"  • Tortuosity: {bell_analysis['tortuosity']:.3f}")
    print(f"  • Spectral Entropy: {bell_analysis['spectral_entropy']:.3f}")
    print(f"  • Errors Detected: {bell_errors}")
    
    # Analyze noisy data
    print("\n🔬 Analyzing noisy data...")
    noisy_tracker = OptimizedCQTTracker()
    for measurement in noisy_measurements:
        noisy_tracker.add_measurement(measurement)
    
    noisy_trajectory = noisy_tracker.get_trajectory()
    noisy_analysis = noisy_tracker.analyze_trajectory()
    noisy_errors = noisy_tracker.detect_errors()
    
    print(f"Noisy Data Results:")
    print(f"  • Trajectory Length: {len(noisy_trajectory)}")
    print(f"  • Mean Magnitude: {np.mean(np.abs(noisy_trajectory)):.3f}")
    print(f"  • Tortuosity: {noisy_analysis['tortuosity']:.3f}")
    print(f"  • Spectral Entropy: {noisy_analysis['spectral_entropy']:.3f}")
    print(f"  • Errors Detected: {noisy_errors}")
    
    # Calculate discrimination metrics
    print("\n📈 Discrimination Analysis:")
    tortuosity_ratio = noisy_analysis['tortuosity'] / bell_analysis['tortuosity']
    entropy_ratio = noisy_analysis['spectral_entropy'] / bell_analysis['spectral_entropy']
    
    print(f"  • Tortuosity Ratio (Noisy/Bell): {tortuosity_ratio:.1f}x")
    print(f"  • Spectral Entropy Ratio (Noisy/Bell): {entropy_ratio:.1f}x")
    
    # Visualization
    print("\n📊 Creating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('CQT Analysis Demo Results', fontsize=16)
    
    # Bell state trajectory
    bell_real = np.real(bell_trajectory)
    bell_imag = np.imag(bell_trajectory)
    axes[0, 0].plot(bell_real, bell_imag, 'b-', alpha=0.7, linewidth=1)
    axes[0, 0].scatter(bell_real[0], bell_imag[0], c='green', s=100, marker='o', label='Start')
    axes[0, 0].scatter(bell_real[-1], bell_imag[-1], c='red', s=100, marker='*', label='End')
    axes[0, 0].set_title('Bell State Trajectory')
    axes[0, 0].set_xlabel('Real Part')
    axes[0, 0].set_ylabel('Imaginary Part')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Noisy data trajectory
    noisy_real = np.real(noisy_trajectory)
    noisy_imag = np.imag(noisy_trajectory)
    axes[0, 1].plot(noisy_real, noisy_imag, 'r-', alpha=0.7, linewidth=1)
    axes[0, 1].scatter(noisy_real[0], noisy_imag[0], c='green', s=100, marker='o', label='Start')
    axes[0, 1].scatter(noisy_real[-1], noisy_imag[-1], c='red', s=100, marker='*', label='End')
    axes[0, 1].set_title('Noisy Data Trajectory')
    axes[0, 1].set_xlabel('Real Part')
    axes[0, 1].set_ylabel('Imaginary Part')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Magnitude evolution
    axes[1, 0].plot(np.abs(bell_trajectory), 'b-', label='Bell State', alpha=0.7)
    axes[1, 0].plot(np.abs(noisy_trajectory), 'r-', label='Noisy Data', alpha=0.7)
    axes[1, 0].set_title('Magnitude Evolution')
    axes[1, 0].set_xlabel('Measurement Index')
    axes[1, 0].set_ylabel('|z|')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Comparison metrics
    metrics = ['Tortuosity', 'Spectral Entropy']
    bell_values = [bell_analysis['tortuosity'], bell_analysis['spectral_entropy']]
    noisy_values = [noisy_analysis['tortuosity'], noisy_analysis['spectral_entropy']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, bell_values, width, label='Bell State', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, noisy_values, width, label='Noisy Data', color='red', alpha=0.7)
    axes[1, 1].set_title('Key Metrics Comparison')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'cqt_demo_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📁 Results saved to: {output_path}")
    
    plt.show()
    
    print("\n✅ Demo completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()