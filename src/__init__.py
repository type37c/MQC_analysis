"""
Complex Quantum Trajectory (CQT) Analysis Framework

This package provides tools for analyzing quantum measurements using complex number
representations, enabling trajectory-based analysis and error detection.

Modules:
- cqt_tracker_v3: Main optimized CQT tracker (recommended)
- cqt_tracker_v2: Improved version with physical constraints
- cqt_tracker: Original implementation (for reference)
- noise_models: Noise simulation utilities
- complex_cqt_operations: Complex mathematical operations
- complex_error_detection: Error detection algorithms
"""

from .cqt_tracker_v3 import OptimizedCQTTracker
from .cqt_tracker_v2 import ImprovedCQTTracker
from .cqt_tracker import CQTMeasurementTracker
from .noise_models import *

__version__ = "1.0.0"
__author__ = "Keisuke Otani"
__email__ = "wacon316@gmail.com"

# Recommended main class for new users
MQCTracker = OptimizedCQTTracker

__all__ = [
    'OptimizedCQTTracker',
    'ImprovedCQTTracker', 
    'CQTMeasurementTracker',
    'MQCTracker'
]