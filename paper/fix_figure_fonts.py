#!/usr/bin/env python3
"""
Fix font issues in Figure 2 and Figure 4 by regenerating them without Japanese characters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add project path
sys.path.append('..')
sys.path.append('../src')

# Import custom modules
from src.cqt_tracker_v3 import OptimizedCQTTracker

# Set matplotlib to use only ASCII fonts
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# High-quality plot settings
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'lines.linewidth': 2,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def fix_w_pattern_figure():
    """Regenerate Figure 2 (W-pattern characteristics) without Japanese text"""
    
    # Load W-pattern analysis results
    results_file = '../w_pattern_detailed_analysis.csv'
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found")
        return
    
    df = pd.read_csv(results_file)
    
    # Separate data types
    bell_data = df[df['name'].str.contains('bell')]
    qv_data = df[df['name'].str.contains('qv_')]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Tortuosity comparison
    ax = axes[0, 0]
    categories = ['Bell States', 'Quantum Volume']
    tortuosity_means = [bell_data['tortuosity'].mean(), qv_data['tortuosity'].mean()]
    tortuosity_stds = [bell_data['tortuosity'].std(), qv_data['tortuosity'].std()]
    
    bars = ax.bar(categories, tortuosity_means, yerr=tortuosity_stds, 
                   capsize=5, color=['skyblue', 'salmon'], alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, tortuosity_means, tortuosity_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 5,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Tortuosity', fontsize=14)
    ax.set_title('(a) Tortuosity Comparison', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(tortuosity_means) * 1.3)
    
    # Add 82x annotation
    ax.annotate('82x difference', xy=(0.5, 100), xytext=(0.5, 120),
                ha='center', fontsize=14, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # 2. Fractal dimension histogram  
    ax = axes[0, 1]
    bell_fractal = bell_data['fractal_dimension'].dropna()
    qv_fractal = qv_data['fractal_dimension'].dropna()
    
    bins = np.linspace(0.16, 0.21, 20)
    ax.hist(bell_fractal, bins=bins, alpha=0.6, label='Bell States', 
            color='skyblue', edgecolor='black', linewidth=1.5)
    ax.hist(qv_fractal, bins=bins, alpha=0.6, label='Quantum Volume', 
            color='salmon', edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Fractal Dimension', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('(b) Fractal Dimension Distribution', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 3. Mean Speed vs Self-intersections
    ax = axes[1, 0]
    ax.scatter(bell_data['mean_speed'], bell_data['self_intersections'], 
               s=100, alpha=0.7, c='skyblue', edgecolors='black', linewidth=1.5,
               label='Bell States')
    ax.scatter(qv_data['mean_speed'], qv_data['self_intersections'], 
               s=100, alpha=0.7, c='salmon', edgecolors='black', linewidth=1.5,
               label='Quantum Volume')
    
    ax.set_xlabel('Mean Speed', fontsize=14)
    ax.set_ylabel('Self-intersections', fontsize=14)
    ax.set_title('(c) Speed vs Complexity', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 4. Self-intersections vs Path Length
    ax = axes[1, 1]
    ax.scatter(bell_data['path_length'], bell_data['self_intersections'], 
               s=100, alpha=0.7, c='skyblue', edgecolors='black', linewidth=1.5,
               label='Bell States')
    ax.scatter(qv_data['path_length'], qv_data['self_intersections'], 
               s=100, alpha=0.7, c='salmon', edgecolors='black', linewidth=1.5,
               label='Quantum Volume')
    
    ax.set_xlabel('Path Length', fontsize=14)
    ax.set_ylabel('Self-intersections', fontsize=14)
    ax.set_title('(d) Path Complexity', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'figures/w_pattern_characteristics_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure 2 saved: {output_file}")
    plt.close()

def fix_spectral_figure():
    """Regenerate Figure 4 (Spectral characteristics) without Japanese text"""
    
    # Load Fourier results
    fourier_file = '../fourier_spectral_analysis_results.csv'
    
    if not os.path.exists(fourier_file):
        print(f"Error: {fourier_file} not found")
        return
        
    df = pd.read_csv(fourier_file)
    
    # Separate data types
    bell_data = df[df['name'].str.contains('bell')]
    qv_data = df[df['name'].str.contains('qv_')]
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Spectral Entropy vs Bandwidth
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(bell_data['spectral_bandwidth'], bell_data['spectral_entropy'],
                s=150, alpha=0.8, c='skyblue', edgecolors='black', linewidth=2,
                label='Bell States', marker='o')
    ax1.scatter(qv_data['spectral_bandwidth'], qv_data['spectral_entropy'],
                s=150, alpha=0.8, c='salmon', edgecolors='black', linewidth=2,
                label='Quantum Volume', marker='s')
    
    ax1.set_xlabel('Spectral Bandwidth (Hz)', fontsize=14)
    ax1.set_ylabel('Spectral Entropy', fontsize=14)
    ax1.set_title('(a) Entropy vs Bandwidth', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add 3.6x annotation
    ax1.annotate('3.6x difference', xy=(0.019, 3.5), xytext=(0.020, 4.2),
                 ha='center', fontsize=12, fontweight='bold', color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # 2. Mean Frequency Distribution
    ax2 = plt.subplot(2, 3, 2)
    bins = np.linspace(0, 0.025, 25)
    ax2.hist(bell_data['mean_frequency'], bins=bins, alpha=0.7, 
             label='Bell States', color='skyblue', edgecolor='black', linewidth=1.5)
    ax2.hist(qv_data['mean_frequency'], bins=bins, alpha=0.7, 
             label='Quantum Volume', color='salmon', edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Mean Frequency (Hz)', fontsize=14)
    ax2.set_ylabel('Count', fontsize=14)
    ax2.set_title('(b) Frequency Distribution', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Peak Power Comparison
    ax3 = plt.subplot(2, 3, 3)
    categories = ['Bell States', 'Quantum Volume']
    peak_means = [bell_data['max_power'].mean(), qv_data['max_power'].mean()]
    peak_stds = [bell_data['max_power'].std(), qv_data['max_power'].std()]
    
    bars = ax3.bar(categories, peak_means, yerr=peak_stds, 
                    capsize=5, color=['skyblue', 'salmon'], alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    
    ax3.set_ylabel('Peak Power', fontsize=14)
    ax3.set_title('(c) Peak Power Analysis', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Spectral Bandwidth vs Spectral Entropy
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(bell_data['spectral_bandwidth'], bell_data['spectral_entropy'],
                s=150, alpha=0.8, c='skyblue', edgecolors='black', linewidth=2,
                label='Bell States', marker='o')
    ax4.scatter(qv_data['spectral_bandwidth'], qv_data['spectral_entropy'],
                s=150, alpha=0.8, c='salmon', edgecolors='black', linewidth=2,
                label='Quantum Volume', marker='s')
    
    ax4.set_xlabel('Spectral Bandwidth (Hz)', fontsize=14)
    ax4.set_ylabel('Spectral Entropy', fontsize=14)
    ax4.set_title('(d) Bandwidth vs Entropy', fontsize=16, fontweight='bold')
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 5. Total Power Distribution
    ax5 = plt.subplot(2, 3, 5)
    data = [bell_data['total_power'].dropna(), 
            qv_data['total_power'].dropna()]
    positions = [1, 2]
    
    bp = ax5.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    
    colors = ['skyblue', 'salmon']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    ax5.set_xticks(positions)
    ax5.set_xticklabels(['Bell States', 'Quantum Volume'], fontsize=12)
    ax5.set_ylabel('Total Power', fontsize=14)
    ax5.set_title('(e) Power Distribution', fontsize=16, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Spectral Centroid
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(range(len(bell_data)), bell_data['spectral_centroid'],
                s=100, alpha=0.8, c='skyblue', edgecolors='black', linewidth=1.5,
                label='Bell States', marker='o')
    ax6.scatter(range(len(bell_data), len(bell_data) + len(qv_data)), 
                qv_data['spectral_centroid'],
                s=100, alpha=0.8, c='salmon', edgecolors='black', linewidth=1.5,
                label='Quantum Volume', marker='s')
    
    ax6.set_xlabel('Sample Index', fontsize=14)
    ax6.set_ylabel('Spectral Centroid (Hz)', fontsize=14)
    ax6.set_title('(f) Spectral Centroid', fontsize=16, fontweight='bold')
    ax6.legend(fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'figures/spectral_characteristics_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure 4 saved: {output_file}")
    plt.close()

if __name__ == "__main__":
    print("Fixing Figure 2 and Figure 4 font issues...")
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Fix both figures
    fix_w_pattern_figure()
    fix_spectral_figure()
    
    print("\nFigures regenerated successfully!")