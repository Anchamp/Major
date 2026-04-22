# Heart Rate Monitor Application

A desktop application that uses computer vision and signal processing to monitor heart rate in real-time from webcam video. The application detects facial features, estimates age and gender, and displays heart rate trends with health status indicators.

## Features

- **Real-time Heart Rate Detection**: Monitors pulse using Fourier transform analysis on forehead region
- **Face Detection**: Automatic detection of facial features using Haar Cascade classifiers
- **Age & Gender Prediction**: AI-powered age and gender estimation using pre-trained Caffe DNN models
- **Heart Rate Status**: Provides health status feedback (Normal, Low, High, etc.) based on age and gender
- **Data Visualization**: 
  - Live heart rate trend graph
  - Pulse signal spectrum display
  - Real-time video feed with annotations
- **Data Logging**: Exports heart rate, age, and gender data to CSV file with timestamps
- **Windowed Averages**: Continuously computes 30‑second sliding averages and logs them to `hr_30sec_avg.csv` for post‑hoc evaluation
- **Modern UI**: PyQt5-based interface with dark theme and intuitive controls

## Project Structure

```
├── main.py                      # Main application (Heart Rate Monitor)
├── user_story_2.py             # Test script for camera and face detection
├── requirements.txt             # Python dependencies
├── deploy_age.prototxt         # Age prediction model architecture
├── age_net.caffemodel          # Pre-trained age prediction weights
├── deploy_gender.prototxt      # Gender prediction model architecture
├── gender_net.caffemodel       # Pre-trained gender prediction weights
├── heart_rate_data.csv         # Output data file (auto-generated)
└── README.md                   # This file
```

## Requirements

### System Requirements
- Python 3.7+
- Webcam/camera device
- OpenCV with DNN module support
- PyQt5
- NumPy

### Dependencies
The application requires the following Python packages:

```
opencv-python>=4.5.0
opencv-contrib-python>=4.5.0
PyQt5>=5.15.0
pyqtgraph>=0.12.0
numpy>=1.19.0
```

## Installation

### 1. Clone/Download the Project
```bash
cd path/to/project
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install numpy opencv-contrib-python PyQt5 pyqtgraph pandas scikit-learn
```

### 4. Verify Model Files
Ensure the following pre-trained model files are in the same directory as main.py:
- `age_net.caffemodel`
- `deploy_age.prototxt`
- `gender_net.caffemodel`
- `deploy_gender.prototxt`

If missing, download from [OpenCV's Model Zoo](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector_models).

## Usage

### Running the Application
```bash
python main.py
```

### Using the Application

1. **Start Monitoring**: Click the "Start" button to begin heart rate monitoring
2. **View Heart Rate**: Monitor real-time heart rate, age, and gender in the main display
3. **View Trends**: Watch the heart rate trend graph update in real-time
4. **View Signal**: Observe the pulse signal spectrum in the frequency domain
5. **Stop Monitoring**: Click "Stop" button to pause monitoring
6. **Data Export**: Heart rate data is automatically saved to `heart_rate_data.csv`

### Heart Rate Status Indicators

The application provides personalized heart rate status based on:
- **Age Group**: Teen, Adult (18-40), Senior (40+)
- **Gender**: Male/Female
- **BPM Range**: Very Low, Low, Normal, High, Very High

### Data Output

Heart rate data is saved to `heart_rate_data.csv` with the following columns:
- `Timestamp`: Date and time of measurement
- `Heart Rate (BPM)`: Beats per minute
- `Age`: Detected age range
- `Gender`: Detected gender

In addition a separate file `hr_30sec_avg.csv` accumulates 30‑second rolling averages with columns:
- `Timestamp`: end time of the window
- `WindowAverageBPM`: average heart rate over the preceding 30 seconds

## How It Works

### Heart Rate Detection Algorithm

Several recent improvements have been made under the hood without altering the user interface:

* Signals are extracted primarily from the **green channel**, where blood-flow related changes are strongest.
* A simple POS‑style chrominance weighting suppresses motion artifacts by subtracting red/blue contributions.
* A median filter is applied to the BPM buffer to remove short-lived spikes (e.g. 104 BPM jumps).
* ECG‑like rolling 30‑second averages are recorded in `hr_30sec_avg.csv` for evaluation by the companion script.

The visualisation pipeline (Eulerian amplification) remains identical to the original design, preserving the forehead blood–flow overlay.

1. **Face Detection**: Uses Haar Cascade classifier to detect face in video frame
2. **ROI Extraction**: Isolates forehead region (most suitable for heart rate detection)
3. **Preprocessing**: 
   - Applies adaptive histogram equalization for better color normalization
   - Normalizes pixel intensities
4. **Gaussian Pyramid**: Builds multi-level Gaussian pyramid for frequency domain analysis
5. **Fourier Transform**: Applies FFT to detect dominant frequency in 1-2 Hz range (60-120 BPM)
6. **Heart Rate Calculation**: Converts dominant frequency to beats per minute
7. **Filtering**: Applies sliding window filter to smooth BPM values

### Age & Gender Prediction

- Uses Caffe deep neural networks pre-trained on facial images
- Models accept 227×227 RGB images as input
- Outputs probability distributions for age and gender categories

### Age Categories
- (0-3), (4-9), (10-15), (16-19), (20-39), (40-59), (60-100)

## Configuration Parameters

Key parameters in `main.py` that can be tuned:

```python
self.bufferSize = 150              # Frames to process
self.minFrequency = 1.0            # Minimum heart rate frequency (Hz)
self.maxFrequency = 2.0            # Maximum heart rate frequency (Hz)
self.bpmCalculationFrequency = 15  # Frames between BPM calculations
self.bpmBufferSize = 10            # Size of BPM smoothing buffer
self.alpha = 170                   # Amplification factor for visualization
```

## Troubleshooting

### Camera Not Opening
- Check if webcam is properly connected
- Verify camera permissions in system settings
- Try using `user_story_2.py` to test camera access

### No Face Detected
- Ensure adequate lighting
- Position face clearly in front of camera
- Keep face within frame

### Inaccurate Heart Rate
- Ensure stable camera position
- Improve lighting conditions
- Keep forehead clearly visible and steady
- Let monitoring run for at least 10 frames for accurate calculation

### Model Loading Errors
- Verify model files are in the correct directory
- Check file permissions
- Ensure model files are not corrupted

## Testing

### Test Camera & Face Detection
Run the test script to verify camera and face detection:
```bash
python user_story_2.py
```

This script will:
- Verify OpenCV installation and version
- Test camera accessibility
- Display face detection in real-time
- Show forehead ROI (Region of Interest)

## Performance Notes

- **Real-time Processing**: 15 FPS video feed for optimal performance
- **GPU Acceleration**: DNN models are CPU-based by default; can be optimized with CUDA
- **Memory Usage**: ~100-150 MB typical operation
- **Data Logging**: CSV file grows ~0.5 KB per minute of monitoring

## Future Enhancements

- [ ] GPU acceleration using CUDA for faster processing
- [ ] Multiple face tracking for group monitoring
- [ ] Heart rate variability (HRV) analysis
- [ ] Export data to cloud services
- [ ] Mobile app integration
- [ ] Real-time alerts for abnormal heart rates
- [ ] Support for multiple camera sources
- [ ] Advanced filtering techniques (Kalman filter)

## Known Limitations

- Single face detection (detects largest face if multiple present)
- Accuracy depends on lighting conditions and camera quality
- Requires continuous visible forehead for reliable measurement
- CPU-intensive processing may limit real-time performance on older systems

## License

[Specify your license here]

## Contributors

[List contributors if applicable]

## Support & Issues

For issues or feature requests, please [create an issue/contact information].

---

**Last Updated**: February 2026

For detailed technical information, refer to the source code comments in `main.py`.
