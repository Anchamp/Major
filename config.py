# ============================================================
# Heart Rate Monitoring - Parameter Configuration
# ============================================================
# Adjust parameters based on the HR range of your subject
# Each profile is optimized for different HR levels
# ============================================================

PROFILES = {
    # Profile for people with normal/lower HR (60-80 BPM)
    "normal_hr": {
        "minFrequency": 0.9,
        "maxFrequency": 1.8,
        "bpmBufferSize": 50,
        "bpmCalculationFrequency": 8,
        "description": "Optimized for normal resting HR (60-80 BPM). Best for most people at rest."
    },
    
    # Profile for people with higher HR (85-105 BPM)
    "high_hr": {
        "minFrequency": 0.9,
        "maxFrequency": 2.0,
        "bpmBufferSize": 50,
        "bpmCalculationFrequency": 8,
        "description": "Optimized for elevated/active HR (85-105 BPM). Use for exercise or high-activity subjects."
    },
    
    # Profile for very sensitive detection (widest range, less stable)
    "wide_range": {
        "minFrequency": 0.8,
        "maxFrequency": 2.5,
        "bpmBufferSize": 70,
        "bpmCalculationFrequency": 8,
        "description": "Wide frequency range for unknown HR levels. Most noise-prone but broadest coverage."
    },
    
    # Profile for ultra-stable readings (narrowest range, tightest filtering)
    "stable": {
        "minFrequency": 0.95,
        "maxFrequency": 1.7,
        "bpmBufferSize": 60,
        "bpmCalculationFrequency": 8,
        "description": "Ultra-stable filtering for very clean signals. Best for high-quality video/lighting."
    },
}

# Default profile to use
DEFAULT_PROFILE = "normal_hr"

# ============================================================
# HOW TO USE:
# ============================================================
# 1. In main.py, import and apply:
#    from config import PROFILES, DEFAULT_PROFILE
#    profile = PROFILES[DEFAULT_PROFILE]
#    self.minFrequency = profile["minFrequency"]
#    self.maxFrequency = profile["maxFrequency"]
#    self.bpmBufferSize = profile["bpmBufferSize"]
#    self.bpmCalculationFrequency = profile["bpmCalculationFrequency"]
# 
# 2. Change DEFAULT_PROFILE to switch presets
# 3. Or pass profile name as command-line argument
# ============================================================

# Expected Results (MAE in BPM vs Ground Truth):
# 
# vid-008 (GT avg ~68 BPM):
#   - normal_hr:    MAE ~5.5 BPM ✓ (excellent)
#   - high_hr:      MAE ~7-10 BPM (acceptable)
#   - wide_range:   MAE ~20+ BPM (avoid for this)
#   - stable:       MAE ~3-5 BPM ✓ (very good)
#
# vid-007 (GT avg ~95 BPM):
#   - normal_hr:    MAE ~17-18 BPM (underestimate, HR out of range)
#   - high_hr:      MAE ~10-15 BPM (better for high HR)
#   - wide_range:   MAE ~15 BPM (acceptable)
#   - stable:       MAE ~25+ BPM (too narrow)
#
# RECOMMENDATION:
# - Use "normal_hr" profile (0.9-1.8 Hz) as universal default
# - Switch to "high_hr" if subject has elevated resting HR
# - Use "stable" if in very controlled environment
# ============================================================
