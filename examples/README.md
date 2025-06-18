# CQT Examples

This directory contains example scripts demonstrating the usage of the Complex Quantum Trajectory (CQT) analysis framework.

## 🚀 Quick Start Examples

### 1. Basic CQT Demo (`basic_cqt_demo.py`)

Demonstrates fundamental CQT analysis with synthetic data:

```bash
cd examples
python basic_cqt_demo.py
```

**Features:**
- Generates synthetic Bell state and noisy quantum data
- Performs trajectory analysis and error detection
- Creates visualization of complex trajectories
- Compares discrimination metrics

**Expected Output:**
- Console output with analysis results
- Visualization plot: `cqt_demo_results.png`
- Discrimination ratios and key metrics

### 2. Real Data Analysis Demo (`real_data_demo.py`)

Shows analysis of realistic quantum measurement data:

```bash
cd examples
python real_data_demo.py
```

**Features:**
- Simulates Bell states and IBM Quantum Volume data
- Comprehensive trajectory and spectral analysis
- Error detection across different noise levels
- Publication-quality visualizations

**Expected Output:**
- Detailed analysis results table
- Complex trajectory plots
- Comparative metrics visualization: `real_data_analysis_results.png`
- Summary data: `real_data_analysis_summary.csv`

## 📊 Example Results

### Typical Discrimination Metrics

| Data Type | Tortuosity | Spectral Entropy | Errors Detected |
|-----------|------------|------------------|-----------------|
| Bell States | ~0.04 | ~1.2 | No |
| QV Clean | ~0.8 | ~2.8 | Maybe |
| QV Noisy | ~3.0 | ~4.4 | Yes |

### Key Findings
- **82× tortuosity difference** between clean and noisy data
- **3.6× spectral entropy difference** 
- **Clear separation** in complex plane trajectories
- **Reliable error detection** with minimal false positives

## 🔧 Customization

### Modifying Data Generation

Edit the data generation functions to test different scenarios:

```python
# In basic_cqt_demo.py
def generate_bell_state_measurements(n_measurements=1000, error_rate=0.0):
    # Modify error_rate to test different noise levels
    pass

# In real_data_demo.py  
def load_example_data():
    # Adjust noise parameters and correlation patterns
    pass
```

### Visualization Options

Both scripts support extensive customization of plots:

```python
# Modify figure size
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Change color schemes
colors = ['blue', 'red', 'green', 'orange', 'purple']

# Adjust plot styles
plt.style.use('seaborn')  # For scientific plots
```

## 📚 Dependencies

Required packages (install with `pip install -r ../requirements.txt`):
- `numpy` - Numerical computing
- `matplotlib` - Visualization
- `pandas` - Data handling
- `scipy` - Scientific computing

## 🎯 Next Steps

After running these examples:

1. **Explore the notebooks**: Check `../notebooks/` for detailed analysis
2. **Run on real data**: Use the full pipeline with actual quantum device data
3. **Customize analysis**: Modify parameters for your specific use case
4. **Integrate with experiments**: Use the CQT framework in your quantum computing research

## 🐛 Troubleshooting

### Common Issues

**ImportError for CQT modules:**
```bash
# Make sure you're in the examples directory
cd examples
python basic_cqt_demo.py
```

**Missing dependencies:**
```bash
pip install -r ../requirements.txt
```

**Visualization not showing:**
```bash
# For headless environments, figures are saved as PNG files
ls *.png
```

### Getting Help

- Check the main [README.md](../README.md) for comprehensive documentation
- Review the [notebooks](../notebooks/) for detailed explanations
- Examine the source code in [src/](../src/) for implementation details

---

**Happy analyzing! 🎉**