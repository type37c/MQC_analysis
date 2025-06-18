# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a CQT (Complex Quantum Trajectory) / MQC (Measurement Quality Complex) research repository implementing practical tools for quantum measurement analysis. The core concept represents quantum measurements as complex numbers: z = (direction) + i*(uncertainty), providing visual trajectory analysis for NISQ-era quantum device diagnostics.

## Key Architecture

### Core Implementation Files
- `src/cqt_tracker.py` - Original CQT implementation (v1) with basic complex number mapping
- `src/cqt_tracker_v2.py` - Improved implementation (v2) addressing physical constraint violations and artificial patterns
- `src/cqt_tracker_v3.py` - Optimized implementation (v3) with realistic detection thresholds and noise-specific strategies
- `src/noise_models.py` - Noise simulation utilities for validation experiments

### Real Data Analysis Scripts
- `real_data_complex_cqt_analysis.py` - Main real data complex CQT analysis
- `w_pattern_detailed_analysis.py` - W-pattern geometric feature extraction
- `complex_error_detection_real_data_simple.py` - Real data error detection
- `fourier_spectral_analysis.py` - Spectral characteristics analysis
- `create_publication_figures.py` - Publication-quality figure generation

### Critical Design Differences Between Versions
- **v1**: Contains artificial temporal modulation (`cos(2π*index/100)`) and allows physical constraint violations (real parts exceeding [-1,+1])
- **v2**: Enforces strict physical constraints using `np.clip()`, removes artificial patterns, includes proper Bell state modeling
- **v3**: Implements realistic detection thresholds, baseline statistical comparison, and noise-type specific detection strategies

### Implementation Philosophy
- **Not using Qiskit**: Deliberately uses NumPy + matplotlib + custom algorithms for lightweight, flexible quantum measurement analysis
- **MQC representation**: Each measurement becomes z = direction + i*uncertainty rather than traditional 0/1 binary outcomes
- **Trajectory analysis**: Focuses on patterns over 1000+ measurements rather than individual results
- **Practical tool focus**: Positioned as complementary diagnostic tool for NISQ devices, not revolutionary theory
- **Real-time capability**: Computational efficiency suitable for hardware monitoring applications
- **Minimal dependencies**: Core functionality requires only numpy, matplotlib, and pandas

## Project Dependencies

### Core Dependencies
```bash
# Install in virtual environment
pip install numpy matplotlib pandas
```

### Optional Dependencies
```bash
# For Jupyter notebooks
pip install jupyter
```

## Common Development Commands

### Setting Up Environment
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install numpy matplotlib pandas jupyter
```

### Running Main Analysis
```bash
# Real data complex CQT analysis (main analysis pipeline)
python3 real_data_complex_cqt_analysis.py

# Publication-quality figure generation
python3 create_publication_figures.py

# Specialized analyses
python3 w_pattern_detailed_analysis.py                    # W-pattern features
python3 complex_error_detection_real_data_simple.py      # Error detection
python3 fourier_spectral_analysis.py                     # Spectral analysis
```

### Testing All Versions
```bash
# Import tests to verify installations
python3 -c "from src.cqt_tracker import CQTMeasurementTracker; print('v1 imported')"
python3 -c "from src.cqt_tracker_v2 import ImprovedCQTTracker; print('v2 imported')"
python3 -c "from src.cqt_tracker_v3 import OptimizedCQTTracker; print('v3 imported')"
```

### Running Specific Experiments
```bash
# Noise model validation
python3 experiments/noise_validation.py

# V3 error detection testing
python3 experiments/test_v3_detection.py

# Error detection debugging
python3 experiments/debug_error_detection.py
```

### Jupyter Notebook Usage
```bash
# Main experiments notebook
jupyter notebook notebooks/01_cqt_basic_experiments.ipynb

# Version comparison analysis
jupyter notebook notebooks/02_cqt_v1_v2_comparison.ipynb

# Real data analysis notebooks
jupyter notebook notebooks/03_cqt_real_data_analysis.ipynb
jupyter notebook notebooks/04_complex_cqt_deep_analysis.ipynb
jupyter notebook notebooks/05_real_data_complex_cqt_analysis_report.ipynb  # Comprehensive report
```

### Paper Compilation (LaTeX)
```bash
# Navigate to paper directory
cd paper/

# Compile LaTeX paper
pdflatex main.tex
pdflatex main.tex  # Run twice for cross-references

# View compiled PDF
# Use your preferred PDF viewer to open main.pdf
```

## Critical Implementation Details

### Physical Constraints (v2+ Implementation)
Real parts must be clipped to [-1, +1] and imaginary parts to [0, 1]:
```python
return np.clip(direction, -1.0, 1.0)  # v2 enforces this, v1 does not
```

### Known Issues in v1
- Bell state |Φ-⟩ produces physically impossible real values like -69.2
- Artificial 100-measurement periodicity from temporal modulation
- False positive error detection for normal quantum states

### Bell State Tracking
Use `BellStateTracker` class (v2 only) for proper two-qubit correlation modeling with correlation analysis via `analyze_correlation()` method.

### Error Detection Strategies
- **v1**: Simple magnitude-based detection (prone to false positives)
- **v2**: State-dependent detection using `improved_error_detection(trajectory, expected_state_type)`
- **v3**: Baseline statistical comparison with realistic thresholds using `OptimizedCQTTracker` class

## Data Flow

1. **Measurement Input**: Quantum measurements (0/1) → Complex number conversion
2. **Trajectory Building**: Complex values → Trajectory storage over time
3. **Statistical Analysis**: `analyze_trajectory()` → coherence, drift, validity metrics
4. **Visualization**: matplotlib complex plane plots and time evolution graphs
5. **Error Detection**: Pattern analysis → Early error identification

## Project Structure

### Source Code (`/src/`)
- Core CQT implementations (v1, v2, v3)
- Noise models and error detection
- Complex operations and analysis runners

### Experiments (`/experiments/`)
- Validation scripts for noise models
- Error detection testing
- Version-specific tests

### Notebooks (`/notebooks/`)
- Research overview and theory (`00_research_overview.md`)
- Experimental implementations and comparisons
- Real data analysis workflows
- Japanese documentation for collaboration

### Data Collection (`/data_collection/`)
- ArXiv paper analysis tools
- Public dataset collection scripts
- Data processing utilities

### Results (`/results/`, `/plots/`)
- Experimental outputs
- Visualization results
- Error detection analysis

### Paper Directory (`/paper/`)
- `main.tex` - Main LaTeX document
- `sections/` - Individual paper sections
- `figures/` - Publication-quality PNG figures (300 DPI)
- `references.bib` - Bibliography database
- `main.pdf` - Compiled paper output

### Documentation (`/docs/`)
- `paper_advice.md` - Critical guidance for paper positioning and revision
- `opus4_analysis.md` - Analysis of implementation issues

## Research Context

This implements practical tools for quantum measurement analysis aimed at:
- Real-time quantum device performance monitoring
- Visual diagnostics for NISQ-era quantum hardware
- Complementary error detection through trajectory pattern analysis
- Hardware characterization and quality assessment

**Important**: Following guidance in `docs/paper_advice.md`, this work is positioned as a practical engineering tool rather than a fundamental theoretical breakthrough. MQC (Measurement Quality Complex) numbers are statistical indicators, not quantum amplitudes.

## Key Results and Validation

### Real Data Performance Metrics
- **82-fold difference** in tortuosity between Bell states (clean) and IBM Quantum Volume data (noisy)
- **3.6-fold difference** in spectral entropy between clean and noisy measurements  
- **0% false positive rate** for Bell state error detection
- **18.6% average error rate** detected in IBM Quantum Volume data
- **Perfect discrimination** between clean and noisy quantum measurements

### Publication Status
- **Paper completed**: "Measurement Quality Complex (MQC) Analysis: A Visual Tool for NISQ-Era Quantum Computing"
- **11-page LaTeX document** with comprehensive experimental validation
- **13 publication-quality figures** (300 DPI) demonstrating quantitative results
- **Ready for arXiv submission** following paper_advice.md recommendations

The repository includes critical analysis from Opus 4.0 in `docs/opus4_analysis.md` identifying fundamental implementation flaws that led to the v2 improvements, and `docs/paper_advice.md` providing essential guidance for proper positioning of the work.