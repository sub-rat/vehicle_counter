import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import sys
import os
from collections import defaultdict
import time
from config import VehicleCountingConfig

class VehicleCounter:
    def __init__(self, video_path, model_path=None):
        self.video_path = video_path
        self.config = VehicleCountingConfig()
        
        # Use config model path or default
        model_path = model_path or self.config.MODEL_PATH
        self.model = YOLO(model_path)
        self.vehicle_classes = self.config.VEHICLE_CLASSES
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize tracking and counting
        self.tracker = sv.ByteTrack()
        self.line_position = None
        self.line_counter = None
        self.vehicle_positions = {}
        self.crossed_vehicles = set()
        self.in_count = 0
        self.out_count = 0
        self.frame_count = 0
        
        # Performance optimization
        self.skip_frames = self.config.SKIP_FRAMES
        self.confidence_threshold = self.config.CONFIDENCE_THRESHOLD
        self.min_vehicle_area = self.config.MIN_VEHICLE_AREA
        
        # Initialize annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.line_annotator = sv.LineZoneAnnotator()
        
        # Analysis results
        self.vehicle_detections = []
        
    def analyze_video_for_line_position(self):
        """Analyze video to find optimal counting line position"""
        print("Analyzing video for optimal counting line position...")
        
        # Sample frames for analysis
        sample_frames = []
        frame_interval = max(1, self.total_frames // self.config.SAMPLE_FRAMES_FOR_ANALYSIS)
        
        for i in range(0, min(self.total_frames, 100), frame_interval):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if ret:
                sample_frames.append(frame)
        
        if not sample_frames:
            # Fallback to middle of frame
            self.line_position = int(self.height * 0.6)
            return
        
        # Analyze vehicle positions in sample frames
        vehicle_y_positions = []
        
        for frame in sample_frames:
            results = self.model(frame, classes=self.vehicle_classes, conf=self.confidence_threshold)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            for bbox in detections.xyxy:
                # Filter by size
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > self.min_vehicle_area:
                    center_y = (bbox[1] + bbox[3]) / 2
                    vehicle_y_positions.append(center_y)
        
        if vehicle_y_positions:
            # Find the most common vehicle position (road level)
            vehicle_y_positions = np.array(vehicle_y_positions)
            # Use histogram to find the most common Y position
            hist, bins = np.histogram(vehicle_y_positions, bins=20)
            peak_idx = np.argmax(hist)
            self.line_position = int(bins[peak_idx])
            
            # Ensure line is not too close to edges
            margin = self.height * self.config.LINE_MARGIN_RATIO
            self.line_position = max(margin, min(self.height - margin, self.line_position))
        else:
            # Fallback to default position
            self.line_position = int(self.height * self.config.DEFAULT_LINE_POSITION_RATIO)
        
        print(f"Optimal counting line position: {self.line_position} (y-coordinate)")
        
        # Initialize line counter
        line_start = sv.Point(0, self.line_position)
        line_end = sv.Point(self.width, self.line_position)
        self.line_counter = sv.LineZone(start=line_start, end=line_end)
        
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def filter_detections(self, detections):
        """Filter detections based on confidence and size"""
        if len(detections) == 0:
            return detections
        
        # Filter by confidence
        confidence_mask = detections.confidence >= self.confidence_threshold
        
        # Filter by bounding box area
        areas = []
        for bbox in detections.xyxy:
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            areas.append(area)
        
        area_mask = np.array(areas) >= self.min_vehicle_area
        
        # Combine masks
        final_mask = confidence_mask & area_mask
        
        # Apply mask to detections
        filtered_detections = detections[final_mask]
        
        return filtered_detections
    
    def count_vehicles(self, detections):
        """Count vehicles crossing the line"""
        if len(detections) == 0:
            return
        
        # Update tracker
        detections = self.tracker.update_with_detections(detections)
        
        # Manual line crossing detection for better accuracy
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is not None:
                bbox = detections.xyxy[i]
                center_y = (bbox[1] + bbox[3]) / 2
                
                if tracker_id in self.vehicle_positions:
                    prev_y = self.vehicle_positions[tracker_id]
                    
                    # Crossing from top to bottom (IN)
                    if prev_y < self.line_position and center_y >= self.line_position:
                        if tracker_id not in self.crossed_vehicles:
                            self.in_count += 1
                            self.crossed_vehicles.add(tracker_id)
                            print(f"Vehicle {tracker_id} crossed IN. Total IN: {self.in_count}")
                    
                    # Crossing from bottom to top (OUT)
                    elif prev_y > self.line_position and center_y <= self.line_position:
                        if tracker_id in self.crossed_vehicles:
                            self.out_count += 1
                            self.crossed_vehicles.remove(tracker_id)
                            print(f"Vehicle {tracker_id} crossed OUT. Total OUT: {self.out_count}")
                
                # Update position
                self.vehicle_positions[tracker_id] = center_y
        
        # Clean up old vehicle positions (prevent memory leaks)
        if len(self.vehicle_positions) > self.config.MAX_TRACKED_VEHICLES:
            current_ids = set(detections.tracker_id) if detections.tracker_id is not None else set()
            self.vehicle_positions = {k: v for k, v in self.vehicle_positions.items() 
                                    if k in current_ids or k in self.crossed_vehicles}
    
    def process_video(self, output_path=None):
        """Process the entire video"""
        # Use config output path if none provided
        if output_path is None:
            output_path = self.config.OUTPUT_PATH
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        if not video_writer.isOpened():
            raise ValueError("Could not initialize video writer")
        
        start_time = time.time()
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            
            self.frame_count += 1
            
            # Skip frames for performance (process every skip_frames+1 frame)
            if self.frame_count % (self.skip_frames + 1) != 0:
                continue
            
            try:
                # Run detection
                results = self.model(frame, classes=self.vehicle_classes)[0]
                detections = sv.Detections.from_ultralytics(results)
                
                # Filter detections
                detections = self.filter_detections(detections)
                
                # Count vehicles
                self.count_vehicles(detections)
                
                # Annotate frame
                annotated_frame = self.annotate_frame(frame, detections)
                
                # Write frame
                video_writer.write(annotated_frame)
                
                # Display progress
                if self.frame_count % (self.fps * self.config.SHOW_PROGRESS_INTERVAL) == 0:
                    elapsed = time.time() - start_time
                    fps_processed = self.frame_count / elapsed
                    print(f"Processed {self.frame_count} frames at {fps_processed:.1f} fps. "
                          f"IN: {self.in_count}, OUT: {self.out_count}, NET: {self.in_count - self.out_count}")
                
                # Optional: Display frame
                if self.config.SHOW_VIDEO:
                    cv2.imshow("Vehicle Counting", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                
            except Exception as e:
                print(f"Error processing frame {self.frame_count}: {e}")
                continue
        
        # Cleanup
        self.cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        
        # Print final results
        processing_time = time.time() - start_time
        print(f"\n=== FINAL RESULTS ===")
        print(f"Total vehicles IN: {self.in_count}")
        print(f"Total vehicles OUT: {self.out_count}")
        print(f"Net vehicle count: {self.in_count - self.out_count}")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Average FPS: {self.frame_count / processing_time:.2f}")
        print(f"Output saved to: {output_path}")
    
    def annotate_frame(self, frame, detections):
        """Annotate frame with detections and counts"""
        # Create labels
        labels = []
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is not None:
                conf = detections.confidence[i]
                labels.append(f"ID:{tracker_id} {conf:.2f}")
            else:
                labels.append("No ID")
        
        # Annotate detections
        annotated_frame = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        # Draw counting line
        cv2.line(annotated_frame, (0, self.line_position), (self.width, self.line_position), 
                self.config.LINE_COLOR, self.config.FONT_THICKNESS)
        
        # Display counts
        net_count = self.in_count - self.out_count
        cv2.putText(annotated_frame, f"Net Count: {net_count}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE, self.config.COUNT_COLOR, self.config.FONT_THICKNESS)
        cv2.putText(annotated_frame, f"In: {self.in_count} | Out: {self.out_count}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.config.IN_OUT_COLOR, self.config.FONT_THICKNESS)
        cv2.putText(annotated_frame, f"Currently Detected: {len([id for id in detections.tracker_id if id is not None])}", 
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.config.DETECTED_COLOR, self.config.FONT_THICKNESS)
        cv2.putText(annotated_frame, f"Frame: {self.frame_count}", (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config.FRAME_COLOR, 1)
        
        return annotated_frame

def main():
    # Get video path from command line arguments or use default
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "./input/video2.mp4"
        print(f"No video path provided. Using default: {video_path}")
    
    try:
        # Initialize vehicle counter
        counter = VehicleCounter(video_path)
        
        # Analyze video for optimal line position
        counter.analyze_video_for_line_position()
        
        # Process video
        counter.process_video()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
