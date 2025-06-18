#!/usr/bin/env python3
"""
Fix Figure 5 - Method Comparison
Change from percentage to qualitative comparison
"""

import numpy as np
import matplotlib.pyplot as plt

# Set up plot style
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'axes.linewidth': 1.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def create_qualitative_comparison_figure():
    """Create method comparison with qualitative ratings instead of percentages"""
    
    # Methods to compare
    methods = ['QPT', 'RB', 'MQC']
    
    # Metrics
    metrics = ['Accuracy', 'Speed', 'Info Content', 'Real-time']
    
    # Qualitative scores (1-5 scale for visualization, but shown as symbols)
    # QPT: High accuracy, Very slow, Very high info, Not real-time
    # RB: Medium accuracy, Fast, Low info, Not real-time  
    # MQC: Good accuracy, Very fast, Good info, Real-time capable
    scores = np.array([
        [5, 1, 5, 1],  # QPT
        [3, 4, 2, 1],  # RB
        [4, 5, 4, 5]   # MQC
    ])
    
    # Colors for methods
    colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    # Rating labels
    rating_labels = ['', 'Low', 'Fair', 'Good', 'High', 'Excellent']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Create bar plot
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, scores[:, idx], color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        
        # Customize y-axis with qualitative labels
        ax.set_ylim(0, 5.5)
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.set_yticklabels(rating_labels)
        
        # Add method labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, fontsize=14, fontweight='bold')
        
        # Add title
        titles = {
            'Accuracy': '(a) Diagnostic Accuracy',
            'Speed': '(b) Computational Speed',
            'Info Content': '(c) Information Richness',
            'Real-time': '(d) Real-time Capability'
        }
        ax.set_title(titles[metric], fontsize=16, fontweight='bold', pad=10)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add qualitative descriptors on bars
        descriptors = {
            'Accuracy': ['Comprehensive', 'Average', 'Reliable'],
            'Speed': ['Hours-Days', 'Minutes', 'Milliseconds'],
            'Info Content': ['Complete', 'Limited', 'Rich'],
            'Real-time': ['No', 'No', 'Yes']
        }
        
        if metric in descriptors:
            for i, (bar, desc) in enumerate(zip(bars, descriptors[metric])):
                height = bar.get_height()
                # Place text above bars
                y_pos = height + 0.1
                if height > 4.5:  # Place inside if bar is too tall
                    y_pos = height - 0.3
                    color = 'white'
                else:
                    color = 'black'
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       desc, ha='center', va='bottom' if height <= 4.5 else 'top',
                       fontsize=11, fontweight='bold', color=color,
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white' if color == 'black' else 'none',
                                edgecolor='none', alpha=0.7 if color == 'black' else 0))
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Add overall comparison text
    fig.text(0.5, 0.02, 
             'QPT: Quantum Process Tomography | RB: Randomized Benchmarking | MQC: Measurement Quality Complex',
             ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    # Save figure
    output_file = 'figures/method_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Qualitative method comparison figure saved: {output_file}")
    plt.close()

if __name__ == "__main__":
    print("Creating qualitative method comparison figure...")
    create_qualitative_comparison_figure()
    print("Done!")