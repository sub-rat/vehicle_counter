# Improved Vehicle Counting System

This is an enhanced vehicle counting system that automatically analyzes videos to find optimal counting line positions and provides accurate vehicle counting with improved performance.

## Features

- **Automatic Line Position Detection**: Analyzes video to find the optimal counting line position
- **Improved Accuracy**: Better tracking and counting logic to prevent double counting
- **Performance Optimization**: Frame skipping and filtering for faster processing
- **Configurable Parameters**: Easy adjustment of settings via `config.py`
- **Memory Management**: Prevents memory leaks with automatic cleanup
- **Progress Tracking**: Real-time progress updates during processing

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download YOLOv8 model (automatically done on first run):
```bash
# The model will be downloaded automatically when you run the script
```

## Usage

### Basic Usage

```bash
# Run with default video
python main.py

# Run with specific video
python main.py ./input/your_video.mp4

# Run with test script
python test_counter.py ./input/your_video.mp4
```

### Advanced Usage

```bash
# Use a different YOLO model
python main.py ./input/video.mp4 yolov8s.pt

# Use test script with custom model
python test_counter.py ./input/video.mp4 yolov8m.pt
```

## Configuration

Edit `config.py` to adjust parameters for your specific use case:

### Detection Settings
- `CONFIDENCE_THRESHOLD`: Minimum confidence for vehicle detection (0.0-1.0)
- `MIN_VEHICLE_AREA`: Minimum bounding box area in pixels
- `VEHICLE_CLASSES`: Classes to detect (car, motorcycle, bus, truck)

### Performance Settings
- `SKIP_FRAMES`: Number of frames to skip for faster processing
- `SAMPLE_FRAMES_FOR_ANALYSIS`: Number of frames to sample for line position analysis

### Line Position Settings
- `DEFAULT_LINE_POSITION_RATIO`: Default line position as ratio of frame height
- `LINE_MARGIN_RATIO`: Minimum margin from frame edges

### Display Settings
- `SHOW_VIDEO`: Set to `True` to display video during processing
- `FONT_SCALE`: Text size for annotations
- `FONT_THICKNESS`: Text thickness for annotations

## How It Works

### 1. Video Analysis
The system first analyzes the video to find the optimal counting line position:
- Samples multiple frames from the video
- Detects vehicles in each frame
- Uses histogram analysis to find the most common vehicle position
- Sets the counting line at this optimal position

### 2. Vehicle Detection and Tracking
- Uses YOLOv8 for vehicle detection
- Applies confidence and size filtering
- Uses ByteTrack for robust vehicle tracking
- Maintains vehicle IDs across frames

### 3. Counting Logic
- Tracks vehicle positions frame by frame
- Detects when vehicles cross the counting line
- Prevents double counting with state management
- Provides separate IN and OUT counts

### 4. Performance Optimization
- Processes every Nth frame (configurable)
- Filters detections by confidence and size
- Cleans up old tracking data to prevent memory leaks
- Provides progress updates during processing

## Output

The system generates:
- **Processed Video**: Annotated video with counting line and vehicle counts
- **Console Output**: Real-time progress and final statistics
- **Final Results**: Total IN/OUT counts and processing performance metrics

## Troubleshooting

### Common Issues

1. **Low Detection Accuracy**
   - Increase `CONFIDENCE_THRESHOLD` in config.py
   - Decrease `MIN_VEHICLE_AREA` for smaller vehicles
   - Use a larger YOLO model (yolov8s.pt, yolov8m.pt, etc.)

2. **Poor Line Position**
   - Adjust `DEFAULT_LINE_POSITION_RATIO` in config.py
   - Check if the video has clear vehicle movement patterns

3. **Slow Processing**
   - Increase `SKIP_FRAMES` in config.py
   - Use a smaller YOLO model (yolov8n.pt)
   - Set `SHOW_VIDEO = False` in config.py

4. **Memory Issues**
   - Decrease `MAX_TRACKED_VEHICLES` in config.py
   - Increase `SKIP_FRAMES` for faster processing

### Performance Tips

- For real-time processing: Use `yolov8n.pt` and increase `SKIP_FRAMES`
- For high accuracy: Use `yolov8x.pt` and decrease `SKIP_FRAMES`
- For memory-constrained systems: Increase `SKIP_FRAMES` and decrease `MAX_TRACKED_VEHICLES`

## File Structure

```
vehicle_count/
├── main.py              # Main vehicle counting script
├── config.py            # Configuration parameters
├── test_counter.py      # Test script for easy usage
├── requirements.txt     # Python dependencies
├── input/               # Input videos
│   ├── video.mp4
│   └── video2.mp4
├── output/              # Output processed videos
└── yolov8n.pt          # YOLO model (downloaded automatically)
```

## Dependencies

- `ultralytics`: YOLOv8 object detection
- `supervision`: Computer vision utilities
- `opencv-python`: Video processing
- `numpy`: Numerical operations

## License

This project is open source and available under the MIT License. 