# Complex Quantum Trajectory (CQT) Theory and Measurement Quality Complex (MQC) Analysis

A novel quantum measurement analysis framework that represents quantum measurements as complex numbers for trajectory-based analysis and error detection.

## 🎯 Overview

This repository implements a practical tool for quantum device monitoring and error detection using **Measurement Quality Complex (MQC)** analysis. Unlike traditional quantum measurement approaches that use binary outcomes (0/1), MQC represents each measurement as a complex number:

```
z = direction + i*uncertainty
```

This enables visualization of quantum measurements as complex trajectories, revealing patterns invisible to traditional analysis methods.

## 🔬 Key Features

- **82-fold discrimination** between clean and noisy quantum data using tortuosity analysis
- **3.6-fold difference** in spectral entropy between Bell states and noisy quantum volume data
- **Real-time error detection** with specialized algorithms for different noise types
- **Visual trajectory analysis** in the complex plane for intuitive understanding
- **Comprehensive spectral analysis** with Fourier-based characterization

## 📊 Scientific Results

Our analysis of real quantum data demonstrates:

- **Bell States (Clean)**: Regular, smooth trajectories with low tortuosity (0.037 ± 0.012)
- **IBM Quantum Volume (Noisy)**: Complex, irregular patterns with high tortuosity (3.03 ± 0.89)
- **Spectral Signatures**: Clear frequency domain separation between clean and noisy data
- **Error Detection**: 0% false positives on Bell states, monotonic error rates for noisy data

## 🏗️ Repository Structure

```
├── src/                          # Core implementation
│   ├── cqt_tracker.py           # Original CQT implementation (v1)
│   ├── cqt_tracker_v2.py        # Improved version with physical constraints
│   ├── cqt_tracker_v3.py        # Optimized version with realistic thresholds
│   ├── noise_models.py          # Noise simulation utilities
│   └── complex_cqt_operations.py # Complex mathematical operations
├── notebooks/                    # Jupyter analysis notebooks
│   ├── 01_cqt_basic_experiments.ipynb
│   ├── 03_cqt_real_data_analysis.ipynb
│   └── 05_real_data_complex_cqt_analysis_report.ipynb
├── experiments/                  # Validation experiments
│   ├── noise_validation.py
│   └── test_v3_detection.py
├── data_collection/             # Real quantum data collection tools
├── paper/                       # Scientific publication (LaTeX)
└── docs/                        # Documentation and technical guides
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy matplotlib scipy pandas jupyter
```

### Basic Usage

```python
from src.cqt_tracker_v3 import OptimizedCQTTracker

# Initialize tracker
tracker = OptimizedCQTTracker()

# Add quantum measurements (0s and 1s)
measurements = [0, 1, 0, 0, 1, 1, 0, ...]
for measurement in measurements:
    tracker.add_measurement(measurement)

# Generate complex trajectory
trajectory = tracker.get_trajectory()

# Analyze trajectory
analysis = tracker.analyze_trajectory()
print(f"Tortuosity: {analysis['tortuosity']:.3f}")
print(f"Spectral Entropy: {analysis['spectral_entropy']:.3f}")

# Detect errors
error_detected = tracker.detect_errors()
print(f"Error Detection: {error_detected}")
```

### Real Data Analysis

```bash
# Analyze real quantum data
python real_data_complex_cqt_analysis.py

# Run comprehensive feature extraction
python w_pattern_detailed_analysis.py

# Perform error detection validation
python complex_error_detection_real_data.py
```

## 📈 Key Results

### Geometric Discrimination
- **Tortuosity**: Bell states (0.037) vs QV data (3.03) - **82× difference**
- **Path Length**: Clear separation between clean and noisy trajectories
- **Curvature Analysis**: Noise signatures in trajectory geometry

### Spectral Analysis
- **Spectral Entropy**: Bell (1.23) vs QV (4.44) - **3.6× difference**
- **Bandwidth**: Noisy data shows broader frequency distribution
- **Dominant Frequencies**: Characteristic patterns for each data type

### Error Detection Performance
- **Bell States**: 0% false positive rate
- **QV Clean**: 8.5% error detection rate
- **QV Moderate**: 15.3% error detection rate  
- **QV Noisy**: 26.4% error detection rate

## 🔬 Jupyter Notebooks

Explore the analysis interactively:

1. **Basic Experiments**: `notebooks/01_cqt_basic_experiments.ipynb`
2. **Real Data Analysis**: `notebooks/03_cqt_real_data_analysis.ipynb`
3. **Comprehensive Report**: `notebooks/05_real_data_complex_cqt_analysis_report.ipynb`

## 📖 Scientific Publication

The complete scientific paper is available in the `paper/` directory:
- **English**: `paper/main.pdf`
- **Japanese**: `paper_japanese/main.pdf`

**Title**: "Measurement Quality Complex (MQC) Analysis: A Visual Tool for NISQ-Era Quantum Computing"

## 🎯 Applications

- **Real-time quantum device monitoring**
- **Hardware characterization and diagnostics**
- **Quantum algorithm performance analysis**
- **Noise pattern identification**
- **Error correction system optimization**

## 🔧 Implementation Versions

### Version 1 (v1): `cqt_tracker.py`
- Original implementation with basic complex mapping
- **Warning**: Contains artificial patterns and physical constraint violations

### Version 2 (v2): `cqt_tracker_v2.py`
- Fixed physical constraints with `np.clip()` enforcement
- Proper Bell state modeling
- Improved error detection algorithms

### Version 3 (v3): `cqt_tracker_v3.py`
- Realistic detection thresholds
- Baseline statistical comparison
- Noise-type specific strategies
- **Recommended for production use**

## 📊 Data Sources

- **Bell States**: Quantum computational basis states (|Φ+⟩, |Φ-⟩, |Ψ+⟩, |Ψ-⟩)
- **IBM Quantum Volume**: Real NISQ device measurement data
- **Synthetic Noise**: Controlled validation experiments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Author**: Keisuke Otani
- **Email**: wacon316@gmail.com
- **Research Focus**: Quantum measurement analysis and NISQ-era diagnostic tools

## 🙏 Acknowledgments

Special thanks to the open-source quantum computing community and the researchers whose work made this analysis possible. This research was conducted using real quantum data from IBM Quantum devices and standard quantum computing frameworks.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{otani2024mqc,
  title={Measurement Quality Complex (MQC) Analysis: A Visual Tool for NISQ-Era Quantum Computing},
  author={Otani, Keisuke},
  year={2024},
  note={Available at: https://github.com/type37c/MQC_analysis}
}
```

---

**Keywords**: Quantum Computing, NISQ, Measurement Analysis, Complex Numbers, Trajectory Analysis, Error Detection, Quantum Noise, Bell States, IBM Quantum
