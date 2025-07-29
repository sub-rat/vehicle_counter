# Vehicle Counting Configuration
# Adjust these parameters based on your video characteristics

class VehicleCountingConfig:
    # Model settings
    MODEL_PATH = "yolov8n.pt"  # or yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection (0.0-1.0)
    MIN_VEHICLE_AREA = 1000  # Minimum bounding box area in pixels
    
    # Performance settings
    SKIP_FRAMES = 2  # Process every (SKIP_FRAMES + 1) frame for speed
    SAMPLE_FRAMES_FOR_ANALYSIS = 20  # Number of frames to sample for line position analysis
    
    # Line position settings
    DEFAULT_LINE_POSITION_RATIO = 0.6  # Default line position as ratio of frame height
    LINE_MARGIN_RATIO = 0.1  # Minimum margin from frame edges
    
    # Tracking settings
    MAX_TRACKED_VEHICLES = 100  # Maximum number of vehicles to track (prevents memory leaks)
    
    # Output settings
    OUTPUT_PATH = "./output/output_vehicle_count.mp4"
    SHOW_PROGRESS_INTERVAL = 5  # Show progress every N seconds
    
    # Display settings
    SHOW_VIDEO = True  # Set to True to display video during processing
    FONT_SCALE = 1.0
    FONT_THICKNESS = 2
    
    # Colors (BGR format)
    LINE_COLOR = (0, 255, 0)  # Green
    COUNT_COLOR = (0, 255, 0)  # Green
    IN_OUT_COLOR = (255, 255, 0)  # Yellow
    DETECTED_COLOR = (0, 255, 255)  # Cyan
    FRAME_COLOR = (255, 255, 255)  # White 