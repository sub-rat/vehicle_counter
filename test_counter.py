#!/usr/bin/env python3
"""
Test script for the improved vehicle counter
"""

import sys
import os
from main import VehicleCounter

def test_video(video_path, model_path=None):
    """Test the vehicle counter with a specific video"""
    print(f"Testing vehicle counter with video: {video_path}")
    print("=" * 50)
    
    try:
        # Initialize counter
        counter = VehicleCounter(video_path, model_path)
        
        # Analyze video for optimal line position
        counter.analyze_video_for_line_position()
        
        # Process video
        counter.process_video()
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return False
    
    return True

def main():
    # Check if video path is provided
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Use default video
        video_path = "./input/video2.mp4"
        print(f"No video path provided. Using default: {video_path}")
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        print("Available videos in input/ directory:")
        if os.path.exists("./input"):
            for file in os.listdir("./input"):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    print(f"  - {file}")
        sys.exit(1)
    
    # Optional model path
    model_path = None
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    
    # Run test
    success = test_video(video_path, model_path)
    
    if success:
        print("\nVehicle counting completed successfully!")
        print("Check the output/ directory for the processed video.")
    else:
        print("\nVehicle counting failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 