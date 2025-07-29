import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import sys

try:
    # Initialize YOLOv8 model (automatically downloads yolov8n.pt if not present)
    model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    sys.exit(1)

# Define class IDs for vehicles (COCO dataset: 2=car, 3=motorcycle, 5=bus, 7=truck)
VEHICLE_CLASSES = [2, 3, 5, 7]

# Get video path from command line arguments or use default
if len(sys.argv) > 1:
    video_path = sys.argv[1]
else:
    video_path = "./input/video2.mp4"  # Default path if none provided
    print(f"No video path provided. Using default: {video_path}")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    sys.exit(1)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer for output
output_path = "./output/output_vehicle_count.mp4"
video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
if not video_writer.isOpened():
    print("Error: Could not initialize video writer")
    cap.release()
    sys.exit(1)

# Define counting line (horizontal line at 70% of frame height)
line_position = int(height * 0.7)
LINE_START = sv.Point(0, line_position)
LINE_END = sv.Point(width, line_position)
line_counter = sv.LineZone(start=LINE_START, end=LINE_END)

# Initialize SORT tracker
tracker = sv.ByteTrack()

# Initialize annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
line_annotator = sv.LineZoneAnnotator()

# Counter for vehicles
vehicle_count = 0
vehicle_positions = {}  # Track previous positions for line crossing detection
in_count = 0
out_count = 0
crossed_vehicles = set()  # Track vehicles that have crossed the line
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    try:
        # Run YOLOv8 detection
        results = model(frame, classes=VEHICLE_CLASSES)[0]
        
        # Convert results to Supervision detections
        detections = sv.Detections.from_ultralytics(results)
        
        # Update tracker
        detections = tracker.update_with_detections(detections)
        
        # Count vehicles crossing the line
        line_counter.trigger(detections)
        
        # Manual line crossing detection
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is not None:
                # Get current position (center of bounding box)
                bbox = detections.xyxy[i]
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Check if vehicle crossed the line
                if tracker_id in vehicle_positions:
                    prev_y = vehicle_positions[tracker_id]
                    
                    # Crossing from top to bottom (IN)
                    if prev_y < line_position and center_y >= line_position:
                        if tracker_id not in crossed_vehicles:
                            in_count += 1
                            crossed_vehicles.add(tracker_id)
                            print(f"Vehicle {tracker_id} crossed IN. Total IN: {in_count}")
                    
                    # Crossing from bottom to top (OUT)
                    elif prev_y > line_position and center_y <= line_position:
                        if tracker_id in crossed_vehicles:
                            out_count += 1
                            crossed_vehicles.remove(tracker_id)
                            print(f"Vehicle {tracker_id} crossed OUT. Total OUT: {out_count}")
                
                # Update position for next frame
                vehicle_positions[tracker_id] = center_y
        
        # Calculate net vehicle count (vehicles currently past the line)
        vehicle_count = in_count - out_count
        
        # Print periodic updates to console
        if frame_count % 30 == 0:  # Every 30 frames (about once per second at 30fps)
            print(f"Frame {frame_count}: IN={in_count}, OUT={out_count}, NET={vehicle_count}, Currently Tracked={len([id for id in detections.tracker_id if id is not None])}")
        
        # Annotate frame
        labels = [f"ID {tracker_id}" if tracker_id is not None else "No ID" for tracker_id in detections.tracker_id]
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = line_annotator.annotate(frame=annotated_frame, line_counter=line_counter)
        
        # Display count on frame
        cv2.putText(annotated_frame, f"Net Count: {vehicle_count}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"In: {in_count} | Out: {out_count}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"Currently Detected: {len([id for id in detections.tracker_id if id is not None])}", (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Write frame to output video
        video_writer.write(annotated_frame)
        
        # Optional: Display frame (comment out if running in a non-GUI environment)
        cv2.imshow("Vehicle Counting", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except Exception as e:
        print(f"Error processing frame: {e}")
        continue

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Final Results:")
print(f"Total vehicles that crossed IN: {in_count}")
print(f"Total vehicles that crossed OUT: {out_count}")
print(f"Net vehicle count: {vehicle_count}")
print(f"Total frames processed: {frame_count}")
