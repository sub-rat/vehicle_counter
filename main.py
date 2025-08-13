import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk
import json
import os
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque
import threading
import time
import csv
from datetime import datetime

class VehicleCountingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Vehicle Counting System")
        self.root.geometry("1400x900")
        
        # Initialize variables
        self.video_path = None
        self.cap = None
        self.model = None
        self.tracker = sv.ByteTrack()
        self.is_playing = False
        self.current_frame = None
        self.frame_count = 0
        self.video_writer = None
        self.output_path = None
        
        # Counting lines and zones (enhanced)
        self.counting_lines = []  # List of counting line objects
        self.counts = defaultdict(lambda: {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0, 'moped': 0, 'total': 0})  # Counts by vehicle type
        self.vehicle_positions = {}  # Current vehicle positions
        self.vehicle_types = {}  # Vehicle type tracking
        self.vehicle_history = defaultdict(list)  # Vehicle crossing history
        self.crossed_vehicles = defaultdict(set)  # Vehicles that crossed each line
        self.line_crossings = defaultdict(list)  # Detailed crossing data
        
        # Multi-line tracking variables
        self.vehicle_paths = defaultdict(list)  # Track vehicle movement paths
        self.vehicle_line_status = defaultdict(dict)  # Track which lines each vehicle has crossed
        self.csv_writer = None
        self.csv_file = None
        self.csv_path = None
        self.real_time_data = []  # Store real-time data for CSV export
        
        # Colors for different lines
        self.line_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
        ]
        
        # GUI setup
        self.setup_gui()
        
        # Load YOLO model with optimized settings
        self.load_yolo_model()
    
    def load_yolo_model(self):
        """Load YOLO model with the selected model type"""
        try:
            model_path = self.model_var.get()
            self.model = YOLO(model_path)
            # Warm up the model
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_frame, verbose=False)
            print(f"YOLO model {model_path} loaded and warmed up successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")
    
    def reload_model(self):
        """Reload the YOLO model with new settings"""
        if self.model:
            del self.model
        self.load_yolo_model()
        messagebox.showinfo("Success", "Model reloaded successfully")
    
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls (scrollable)
        left_panel_container = ttk.Frame(main_frame, width=350)
        left_panel_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Create canvas and scrollbar for left panel
        left_canvas = tk.Canvas(left_panel_container, width=350)
        left_scrollbar = ttk.Scrollbar(left_panel_container, orient="vertical", command=left_canvas.yview)
        left_panel = ttk.Frame(left_canvas)
        
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create window in canvas
        left_canvas.create_window((0, 0), window=left_panel, anchor="nw")
        
        # Configure scrolling
        left_panel.bind("<Configure>", lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.bind("<Configure>", lambda e: left_canvas.itemconfig(left_canvas.find_withtag("window"), width=e.width))
        
        # Add mouse wheel scrolling
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Bind mouse wheel to the left panel container
        left_panel_container.bind("<Enter>", lambda e: left_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        left_panel_container.bind("<Leave>", lambda e: left_canvas.unbind_all("<MouseWheel>"))
        
        # Right panel for video display
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Video canvas
        self.canvas = tk.Canvas(right_panel, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Control buttons
        ttk.Button(left_panel, text="Load Video", command=self.load_video).pack(fill=tk.X, pady=5)
        ttk.Button(left_panel, text="Start Processing", command=self.start_processing).pack(fill=tk.X, pady=5)
        ttk.Button(left_panel, text="Stop Processing", command=self.stop_processing).pack(fill=tk.X, pady=5)
        ttk.Button(left_panel, text="Clear Lines", command=self.clear_lines).pack(fill=tk.X, pady=5)
        ttk.Button(left_panel, text="Save Configuration", command=self.save_config).pack(fill=tk.X, pady=5)
        ttk.Button(left_panel, text="Load Configuration", command=self.load_config).pack(fill=tk.X, pady=5)
        ttk.Button(left_panel, text="Export Data", command=self.export_data).pack(fill=tk.X, pady=5)
        ttk.Button(left_panel, text="Save Processed Video", command=self.save_processed_video).pack(fill=tk.X, pady=5)
        ttk.Button(left_panel, text="Start CSV Logging", command=self.start_csv_logging).pack(fill=tk.X, pady=5)
        ttk.Button(left_panel, text="Stop CSV Logging", command=self.stop_csv_logging).pack(fill=tk.X, pady=5)
        
        # Performance settings
        perf_frame = ttk.LabelFrame(left_panel, text="Performance Settings")
        perf_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(perf_frame, text="Display Update Rate:").pack(anchor=tk.W)
        self.display_rate_var = tk.IntVar(value=5)
        display_spinbox = ttk.Spinbox(perf_frame, from_=1, to=30, textvariable=self.display_rate_var, width=10)
        display_spinbox.pack(fill=tk.X, pady=2)
        
        ttk.Label(perf_frame, text="Confidence Threshold:").pack(anchor=tk.W, pady=(10, 0))
        self.conf_threshold_var = tk.DoubleVar(value=0.2)
        conf_spinbox = ttk.Spinbox(perf_frame, from_=0.05, to=0.9, increment=0.05, textvariable=self.conf_threshold_var, width=10)
        conf_spinbox.pack(fill=tk.X, pady=2)
        
        ttk.Label(perf_frame, text="Display FPS:").pack(anchor=tk.W, pady=(10, 0))
        self.display_fps_var = tk.IntVar(value=30)
        fps_spinbox = ttk.Spinbox(perf_frame, from_=15, to=60, textvariable=self.display_fps_var, width=10)
        fps_spinbox.pack(fill=tk.X, pady=2)
        
        ttk.Label(perf_frame, text="Processing Speed:").pack(anchor=tk.W, pady=(10, 0))
        self.speed_var = tk.StringVar(value="Ultra Fast")
        speed_combo = ttk.Combobox(perf_frame, textvariable=self.speed_var, values=["Ultra Fast", "Fast", "Normal", "Accurate"], state="readonly", width=10)
        speed_combo.pack(fill=tk.X, pady=2)
        
        # Add note about video speed
        ttk.Label(perf_frame, text="Note: Video plays at original speed", 
                 font=("Arial", 8), foreground="gray").pack(pady=2)
        
        # Multi-line tracking settings
        tracking_frame = ttk.LabelFrame(left_panel, text="Multi-Line Tracking")
        tracking_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(tracking_frame, text="Path Tracking:").pack(anchor=tk.W)
        self.path_tracking_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tracking_frame, text="Enable Vehicle Path Tracking", variable=self.path_tracking_var).pack(anchor=tk.W)
        
        ttk.Label(tracking_frame, text="CSV Logging:").pack(anchor=tk.W, pady=(10, 0))
        self.csv_logging_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tracking_frame, text="Enable Real-time CSV Logging", variable=self.csv_logging_var).pack(anchor=tk.W)
        
        ttk.Label(tracking_frame, text="Path History Length:").pack(anchor=tk.W, pady=(10, 0))
        self.path_history_var = tk.IntVar(value=50)
        path_spinbox = ttk.Spinbox(tracking_frame, from_=10, to=200, textvariable=self.path_history_var, width=10)
        path_spinbox.pack(fill=tk.X, pady=2)
        
        ttk.Label(perf_frame, text="Min Vehicle Size:").pack(anchor=tk.W, pady=(10, 0))
        self.min_size_var = tk.IntVar(value=100)
        size_spinbox = ttk.Spinbox(perf_frame, from_=50, to=500, increment=25, textvariable=self.min_size_var, width=10)
        size_spinbox.pack(fill=tk.X, pady=2)
        
        ttk.Label(perf_frame, text="Model Type:").pack(anchor=tk.W, pady=(10, 0))
        self.model_var = tk.StringVar(value="yolov8n.pt")
        model_combo = ttk.Combobox(perf_frame, textvariable=self.model_var, 
                                  values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"], 
                                  state="readonly", width=10)
        model_combo.pack(fill=tk.X, pady=2)
        
        ttk.Button(perf_frame, text="Reload Model", command=self.reload_model).pack(fill=tk.X, pady=5)
        
        # Line management
        line_frame = ttk.LabelFrame(left_panel, text="Counting Lines")
        line_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(line_frame, text="Line Name:").pack(anchor=tk.W)
        self.line_name_var = tk.StringVar()
        ttk.Entry(line_frame, textvariable=self.line_name_var).pack(fill=tk.X, pady=2)
        
        ttk.Label(line_frame, text="Line Color:").pack(anchor=tk.W, pady=(10, 0))
        self.color_button = ttk.Button(line_frame, text="Choose Color", command=self.choose_color)
        self.color_button.pack(fill=tk.X, pady=2)
        
        self.current_color = (255, 0, 0)  # Default red
        
        # Lines list
        ttk.Label(line_frame, text="Active Lines:").pack(anchor=tk.W, pady=(10, 0))
        self.lines_listbox = tk.Listbox(line_frame, height=6)
        self.lines_listbox.pack(fill=tk.X, pady=2)
        
        ttk.Button(line_frame, text="Delete Selected Line", command=self.delete_line).pack(fill=tk.X, pady=2)
        
        # Statistics
        stats_frame = ttk.LabelFrame(left_panel, text="Statistics")
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Create notebook for tabs
        self.stats_notebook = ttk.Notebook(stats_frame)
        self.stats_notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Summary tab
        summary_frame = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(summary_frame, text="Summary")
        self.summary_text = tk.Text(summary_frame, height=10, width=40, font=('Consolas', 9))
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
        # Detailed tab
        detailed_frame = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(detailed_frame, text="Detailed")
        self.detailed_text = tk.Text(detailed_frame, height=10, width=40, font=('Consolas', 9))
        self.detailed_text.pack(fill=tk.BOTH, expand=True)
        
        # Analytics tab
        analytics_frame = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(analytics_frame, text="Analytics")
        self.analytics_text = tk.Text(analytics_frame, height=10, width=40, font=('Consolas', 9))
        self.analytics_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Drawing variables
        self.drawing = False
        self.start_point = None
        self.current_line = None
    
    def choose_color(self):
        color = colorchooser.askcolor(title="Choose Line Color")
        if color[0]:
            self.current_color = tuple(int(c) for c in color[0])
            self.color_button.configure(text=f"Color: RGB{self.current_color}")
    
    def start_csv_logging(self):
        """Start real-time CSV logging"""
        if not self.counting_lines:
            messagebox.showwarning("Warning", "Please add counting lines first")
            return
        
        # Create output directory
        os.makedirs("./output", exist_ok=True)
        
        # Create CSV file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"./output/vehicle_tracking_{timestamp}.csv"
        
        try:
            self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write CSV header
            header = [
                'Time', 'Frame', 'Vehicle_ID', 'Vehicle_Type', 'Confidence',
                'Center_X', 'Center_Y', 'Bbox_X1', 'Bbox_Y1', 'Bbox_X2', 'Bbox_Y2',
                'Line_Crossed', 'Direction', 'From_Line', 'To_Line', 'Path_Length',
                'Lines_Crossed_Count', 'Current_Speed', 'Total_Distance'
            ]
            self.csv_writer.writerow(header)
            self.csv_file.flush()
            
            self.csv_logging_var.set(True)
            self.status_var.set(f"CSV logging started: {os.path.basename(self.csv_path)}")
            messagebox.showinfo("Success", f"CSV logging started\nFile: {self.csv_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start CSV logging: {e}")
    
    def stop_csv_logging(self):
        """Stop real-time CSV logging"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            self.csv_logging_var.set(False)
            self.status_var.set("CSV logging stopped")
            messagebox.showinfo("Success", f"CSV logging stopped\nData saved to: {self.csv_path}")
    
    def log_vehicle_data(self, frame_num, tracker_id, vehicle_type, confidence, 
                        center_x, center_y, bbox, line_crossed=None, direction=None, 
                        from_line=None, to_line=None):
        """Log vehicle data to CSV in real-time"""
        if not self.csv_writer or not self.csv_logging_var.get():
            return
        
        try:
            # Calculate additional metrics (optimized for speed)
            path_length = len(self.vehicle_paths.get(tracker_id, []))
            lines_crossed_count = len(self.vehicle_line_status.get(tracker_id, {}))
            
            # Calculate current speed (distance from previous position) - simplified
            current_speed = 0
            total_distance = 0
            if tracker_id in self.vehicle_paths and len(self.vehicle_paths[tracker_id]) > 1:
                prev_pos = self.vehicle_paths[tracker_id][-2]
                current_speed = np.sqrt((center_x - prev_pos[0])**2 + (center_y - prev_pos[1])**2)
                
                # Simplified total distance calculation (only last 10 points for speed)
                path_points = self.vehicle_paths[tracker_id][-10:] if len(self.vehicle_paths[tracker_id]) > 10 else self.vehicle_paths[tracker_id]
                for i in range(1, len(path_points)):
                    prev = path_points[i-1]
                    curr = path_points[i]
                    total_distance += np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
            
            # Write data to CSV (optimized format)
            row = [
                datetime.now().strftime("%H:%M:%S.%f")[:-3],  # Shorter timestamp
                frame_num,
                tracker_id,
                vehicle_type,
                f"{confidence:.2f}",  # Reduced precision
                f"{center_x:.0f}",    # Integer coordinates for speed
                f"{center_y:.0f}",
                f"{bbox[0]:.0f}",
                f"{bbox[1]:.0f}",
                f"{bbox[2]:.0f}",
                f"{bbox[3]:.0f}",
                line_crossed or "",
                direction or "",
                from_line or "",
                to_line or "",
                path_length,
                lines_crossed_count,
                f"{current_speed:.0f}",  # Integer speed
                f"{total_distance:.0f}"  # Integer distance
            ]
            
            self.csv_writer.writerow(row)
            
            # Flush every 10 rows for better performance
            if frame_num % 10 == 0:
                self.csv_file.flush()
            
        except Exception as e:
            print(f"Error logging to CSV: {e}")
    
    def load_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(file_path)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video file")
                return
            
            # Get video properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Load first frame
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                self.status_var.set(f"Video loaded: {os.path.basename(file_path)} ({self.width}x{self.height})")
            
            # Reset video to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def display_frame(self, frame):
        if frame is None:
            return
        
        # Resize frame to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Calculate scaling to fit frame in canvas
            frame_height, frame_width = frame.shape[:2]
            scale_x = canvas_width / frame_width
            scale_y = canvas_height / frame_height
            scale = min(scale_x, scale_y)
            
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            
            # Resize frame with better interpolation for smoother display
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Convert to PIL Image
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # Update canvas efficiently
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo, anchor=tk.CENTER)
            
            # Draw counting lines
            self.draw_counting_lines(scale)
    
    def draw_counting_lines(self, scale=1.0):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        for i, line in enumerate(self.counting_lines):
            x1, y1, x2, y2 = line['coords']
            color = line['color']
            name = line['name']
            
            # Scale coordinates
            scaled_x1 = int(x1 * scale) + canvas_width//2 - int(self.width * scale)//2
            scaled_y1 = int(y1 * scale) + canvas_height//2 - int(self.height * scale)//2
            scaled_x2 = int(x2 * scale) + canvas_width//2 - int(self.width * scale)//2
            scaled_y2 = int(y2 * scale) + canvas_height//2 - int(self.height * scale)//2
            
            # Draw line
            self.canvas.create_line(scaled_x1, scaled_y1, scaled_x2, scaled_y2, 
                                  fill=f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}', 
                                  width=3, tags=f"line_{i}")
            
            # Draw line name
            self.canvas.create_text(scaled_x1, scaled_y1 - 20, text=name, 
                                  fill=f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}', 
                                  font=('Arial', 12, 'bold'), tags=f"line_{i}")
    
    def on_canvas_click(self, event):
        if not self.cap:
            return
        
        self.drawing = True
        
        # Get canvas and frame dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        scale_x = canvas_width / self.width
        scale_y = canvas_height / self.height
        scale = min(scale_x, scale_y)
        
        # Convert canvas coordinates to frame coordinates
        frame_x = (event.x - canvas_width//2 + int(self.width * scale)//2) / scale
        frame_y = (event.y - canvas_height//2 + int(self.height * scale)//2) / scale
        
        # Ensure coordinates are within frame bounds
        frame_x = max(0, min(self.width, int(frame_x)))
        frame_y = max(0, min(self.height, int(frame_y)))
        
        self.start_point = (frame_x, frame_y)
    
    def on_canvas_drag(self, event):
        if not self.drawing or not self.start_point:
            return
        
        # Get canvas and frame dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        scale_x = canvas_width / self.width
        scale_y = canvas_height / self.height
        scale = min(scale_x, scale_y)
        
        # Convert canvas coordinates to frame coordinates
        frame_x = (event.x - canvas_width//2 + int(self.width * scale)//2) / scale
        frame_y = (event.y - canvas_height//2 + int(self.height * scale)//2) / scale
        
        # Ensure coordinates are within frame bounds
        frame_x = max(0, min(self.width, int(frame_x)))
        frame_y = max(0, min(self.height, int(frame_y)))
        
        end_point = (frame_x, frame_y)
        
        # Remove previous temporary line
        self.canvas.delete("temp_line")
        
        # Draw temporary line
        scaled_x1 = int(self.start_point[0] * scale) + canvas_width//2 - int(self.width * scale)//2
        scaled_y1 = int(self.start_point[1] * scale) + canvas_height//2 - int(self.height * scale)//2
        scaled_x2 = int(end_point[0] * scale) + canvas_width//2 - int(self.width * scale)//2
        scaled_y2 = int(end_point[1] * scale) + canvas_height//2 - int(self.height * scale)//2
        
        self.canvas.create_line(scaled_x1, scaled_y1, scaled_x2, scaled_y2, 
                              fill=f'#{self.current_color[0]:02x}{self.current_color[1]:02x}{self.current_color[2]:02x}', 
                              width=3, tags="temp_line")
    
    def on_canvas_release(self, event):
        if not self.drawing or not self.start_point:
            return
        
        self.drawing = False
        
        # Get canvas and frame dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        scale_x = canvas_width / self.width
        scale_y = canvas_height / self.height
        scale = min(scale_x, scale_y)
        
        # Convert canvas coordinates to frame coordinates
        frame_x = (event.x - canvas_width//2 + int(self.width * scale)//2) / scale
        frame_y = (event.y - canvas_height//2 + int(self.height * scale)//2) / scale
        
        # Ensure coordinates are within frame bounds
        frame_x = max(0, min(self.width, int(frame_x)))
        frame_y = max(0, min(self.height, int(frame_y)))
        
        end_point = (frame_x, frame_y)
        
        # Remove temporary line
        self.canvas.delete("temp_line")
        
        # Add new counting line
        line_name = self.line_name_var.get() or f"Line {len(self.counting_lines) + 1}"
        
        new_line = {
            'name': line_name,
            'coords': (self.start_point[0], self.start_point[1], end_point[0], end_point[1]),
            'color': self.current_color
        }
        
        self.counting_lines.append(new_line)
        self.update_lines_list()
        self.draw_counting_lines()
        
        # Clear line name entry
        self.line_name_var.set("")
        
        self.start_point = None
    
    def update_lines_list(self):
        self.lines_listbox.delete(0, tk.END)
        for line in self.counting_lines:
            self.lines_listbox.insert(tk.END, line['name'])
    
    def delete_line(self):
        selection = self.lines_listbox.curselection()
        if selection:
            index = selection[0]
            del self.counting_lines[index]
            self.update_lines_list()
            self.draw_counting_lines()
    
    def clear_lines(self):
        self.counting_lines.clear()
        self.update_lines_list()
        self.canvas.delete("all")
        if hasattr(self, 'photo'):
            self.canvas.create_image(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2, 
                                   image=self.photo, anchor=tk.CENTER)
    
    def save_config(self):
        if not self.counting_lines:
            messagebox.showwarning("Warning", "No counting lines to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            config = {
                'video_path': self.video_path,
                'counting_lines': self.counting_lines,
                'video_properties': {
                    'width': self.width,
                    'height': self.height,
                    'fps': self.fps
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            messagebox.showinfo("Success", "Configuration saved successfully")
    
    def load_config(self):
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                
                self.counting_lines = config['counting_lines']
                self.update_lines_list()
                
                # Load video if path is different
                if config.get('video_path') and config['video_path'] != self.video_path:
                    self.video_path = config['video_path']
                    self.cap = cv2.VideoCapture(self.video_path)
                    if self.cap.isOpened():
                        ret, frame = self.cap.read()
                        if ret:
                            self.display_frame(frame)
                
                self.draw_counting_lines()
                messagebox.showinfo("Success", "Configuration loaded successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
    
    def start_processing(self):
        if not self.cap or not self.counting_lines:
            messagebox.showwarning("Warning", "Please load a video and add counting lines first")
            return
        
        if self.is_playing:
            return
        
        # Create output directory
        os.makedirs("./output", exist_ok=True)
        
        # Initialize video writer
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_path = f"./output/processed_video_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        if not self.video_writer.isOpened():
            messagebox.showerror("Error", "Could not initialize video writer")
            return
        
        # Auto-start CSV logging if enabled
        if self.csv_logging_var.get() and not self.csv_writer:
            self.start_csv_logging()
        
        self.is_playing = True
        self.status_var.set("Processing at 2x speed...")
        
        # Start processing in separate thread
        self.processing_thread = threading.Thread(target=self.process_video)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_processing(self):
        self.is_playing = False
        self.status_var.set("Stopped")
        
        # Close video writer
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print(f"Processed video saved to: {self.output_path}")
    
    def process_video(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Performance optimization settings
        display_rate = self.display_rate_var.get()  # Get from GUI
        display_fps = self.display_fps_var.get()  # Get from GUI
        speed_mode = self.speed_var.get()  # Get speed mode
        min_size = self.min_size_var.get()  # Get minimum vehicle size
        
        # Video playback settings - play at 2x speed
        original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        target_fps = original_fps * 2  # 2x speed
        frame_delay = 1.0 / target_fps if target_fps > 0 else 1.0 / 60.0  # 2x faster than original
        
        # Adjust settings based on speed mode - optimized for 2x speed
        if speed_mode == "Ultra Fast":
            skip_frames = 0  # Process every frame
            conf_threshold = 0.15
            min_area = min_size
            model_size = 416  # Smaller input size for maximum speed
            display_skip = 2  # Update display every 2 frames
        elif speed_mode == "Fast":
            skip_frames = 0  # Process every frame
            conf_threshold = 0.2
            min_area = min_size
            model_size = 480  # Smaller input size for speed
            display_skip = 3  # Update display every 3 frames
        elif speed_mode == "Normal":
            skip_frames = 0  # Process every frame
            conf_threshold = 0.3
            min_area = min_size
            model_size = 640
            display_skip = 4  # Update display every 4 frames
        else:  # Accurate
            skip_frames = 0  # Process every frame
            conf_threshold = 0.4
            min_area = min_size
            model_size = 640
            display_skip = 5  # Update display every 5 frames
        
        frame_interval = 0
        last_display_time = time.time()
        processed_frames = 0
        display_interval = max(1, 60 // display_fps)  # Calculate display interval
        
        while self.is_playing:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            self.frame_count += 1
            frame_interval += 1
            
            # Process every frame for maximum accuracy
            # No frame skipping - process all frames
            
            processed_frames += 1
            
            # Run detection with optimized settings for speed
            results = self.model(frame, classes=[2, 3, 5, 7], conf=conf_threshold, iou=0.4, imgsz=model_size, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # Filter detections by size for better accuracy
            if len(detections) > 0:
                areas = []
                for bbox in detections.xyxy:
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    areas.append(area)
                
                # Filter out very small detections
                area_mask = np.array(areas) >= min_area
                detections = detections[area_mask]
            
            # Update tracker
            detections = self.tracker.update_with_detections(detections)
            
            # Process vehicle counting
            self.process_vehicle_counting(frame, detections)
            
            # Optimized display update - less frequent updates for better performance
            if processed_frames % display_skip == 0:
                self.root.after(0, self.update_display, frame, detections)
            
            # Write annotated frame to output video (not the original frame)
            if self.video_writer and self.video_writer.isOpened():
                # Create annotated frame for output video
                output_frame = self.create_annotated_frame(frame, detections)
                self.video_writer.write(output_frame)
            
            # Maintain original video speed for playback
            time.sleep(frame_delay)
    
    def process_vehicle_counting(self, frame, detections):
        if len(detections) == 0:
            return
        
        # Vehicle class mapping
        class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        # Process each vehicle (enhanced with path tracking)
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is not None:
                bbox = detections.xyxy[i]
                class_id = int(detections.class_id[i]) if detections.class_id is not None else 2
                confidence = detections.confidence[i] if detections.confidence is not None else 0.0
                
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Determine vehicle type
                vehicle_type = class_names.get(class_id, 'car')
                if vehicle_type == 'motorcycle' and (bbox[2] - bbox[0]) < 50:  # Small motorcycles
                    vehicle_type = 'moped'
                
                # Update vehicle information
                self.vehicle_positions[tracker_id] = (center_x, center_y)
                self.vehicle_types[tracker_id] = vehicle_type
                
                # Track vehicle path if enabled (optimized for speed)
                if self.path_tracking_var.get():
                    if tracker_id not in self.vehicle_paths:
                        self.vehicle_paths[tracker_id] = []
                    
                    # Add current position to path (every 2nd frame for speed)
                    if self.frame_count % 2 == 0:
                        self.vehicle_paths[tracker_id].append((center_x, center_y))
                        
                        # Limit path history to prevent memory issues
                        max_path_length = self.path_history_var.get()
                        if len(self.vehicle_paths[tracker_id]) > max_path_length:
                            self.vehicle_paths[tracker_id] = self.vehicle_paths[tracker_id][-max_path_length:]
                
                # Check line crossings with enhanced tracking
                line_crossed, direction, from_line, to_line = self.check_line_crossings(tracker_id, center_x, center_y, vehicle_type, confidence)
                
                # Log vehicle data to CSV in real-time
                self.log_vehicle_data(
                    self.frame_count, tracker_id, vehicle_type, confidence,
                    center_x, center_y, bbox, line_crossed, direction, from_line, to_line
                )
    
    def check_line_crossings(self, tracker_id, center_x, center_y, vehicle_type, confidence):
        if tracker_id not in self.vehicle_positions:
            return None, None, None, None
        
        prev_x, prev_y = self.vehicle_positions[tracker_id]
        line_crossed = None
        direction = None
        from_line = None
        to_line = None
        
        # Initialize vehicle line status if not exists
        if tracker_id not in self.vehicle_line_status:
            self.vehicle_line_status[tracker_id] = {}
        
        for i, line in enumerate(self.counting_lines):
            x1, y1, x2, y2 = line['coords']
            line_name = line['name']
            
            # Check if vehicle crossed this line
            if self.line_crossed(prev_x, prev_y, center_x, center_y, x1, y1, x2, y2):
                if tracker_id not in self.crossed_vehicles[i]:
                    # Vehicle crossed line going forward (IN)
                    self.counts[line_name][vehicle_type] += 1
                    self.counts[line_name]['total'] += 1
                    self.crossed_vehicles[i].add(tracker_id)
                    
                    # Update vehicle line status
                    self.vehicle_line_status[tracker_id][line_name] = 'IN'
                    
                    # Determine from_line (previous line crossed)
                    from_line = self.get_previous_line_crossed(tracker_id, line_name)
                    to_line = line_name
                    
                    # Record crossing details
                    crossing_data = {
                        'timestamp': time.time(),
                        'vehicle_id': tracker_id,
                        'vehicle_type': vehicle_type,
                        'confidence': confidence,
                        'direction': 'IN',
                        'line_name': line_name,
                        'from_line': from_line,
                        'to_line': to_line
                    }
                    self.line_crossings[line_name].append(crossing_data)
                    self.vehicle_history[tracker_id].append(crossing_data)
                    
                    line_crossed = line_name
                    direction = 'IN'
                    
                    print(f"Vehicle {tracker_id} ({vehicle_type}) crossed {line_name} IN (from {from_line or 'start'})")
                else:
                    # Vehicle crossed line going backward (OUT)
                    self.counts[line_name][vehicle_type] += 1
                    self.counts[line_name]['total'] += 1
                    self.crossed_vehicles[i].discard(tracker_id)
                    
                    # Update vehicle line status
                    if line_name in self.vehicle_line_status[tracker_id]:
                        del self.vehicle_line_status[tracker_id][line_name]
                    
                    # Determine from_line and to_line
                    from_line = line_name
                    to_line = self.get_next_line_crossed(tracker_id, line_name)
                    
                    # Record crossing details
                    crossing_data = {
                        'timestamp': time.time(),
                        'vehicle_id': tracker_id,
                        'vehicle_type': vehicle_type,
                        'confidence': confidence,
                        'direction': 'OUT',
                        'line_name': line_name,
                        'from_line': from_line,
                        'to_line': to_line
                    }
                    self.line_crossings[line_name].append(crossing_data)
                    self.vehicle_history[tracker_id].append(crossing_data)
                    
                    line_crossed = line_name
                    direction = 'OUT'
                    
                    print(f"Vehicle {tracker_id} ({vehicle_type}) crossed {line_name} OUT (to {to_line or 'end'})")
        
        return line_crossed, direction, from_line, to_line
    
    def create_annotated_frame(self, frame, detections):
        """Create annotated frame for output video with all tracking data"""
        if frame is None:
            return frame
        
        # Create a copy of the frame for annotation
        annotated_frame = frame.copy()
        
        # Draw vehicle detections with professional bounding boxes
        if detections is not None and len(detections) > 0:
            for i, tracker_id in enumerate(detections.tracker_id):
                if tracker_id is not None:
                    bbox = detections.xyxy[i]
                    class_id = int(detections.class_id[i]) if detections.class_id is not None else 2
                    confidence = detections.confidence[i] if detections.confidence is not None else 0.0
                    
                    # Get vehicle type
                    vehicle_type = self.vehicle_types.get(tracker_id, 'car')
                    
                    # Professional color scheme
                    type_colors = {
                        'car': (0, 150, 255),      # Orange
                        'motorcycle': (255, 50, 50),   # Red
                        'bus': (255, 0, 255),      # Magenta
                        'truck': (0, 255, 255),    # Cyan
                        'moped': (255, 255, 0)     # Yellow
                    }
                    color = type_colors.get(vehicle_type, (0, 150, 255))
                    
                    # Draw professional bounding box
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Draw filled rectangle with transparency effect
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
                    
                    # Draw border
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw professional label with background
                    label = f"{tracker_id} | {vehicle_type.upper()} | {confidence:.1f}"
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Draw label background
                    cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), 
                                (x1 + label_width + 10, y1), color, -1)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), 
                                (x1 + label_width + 10, y1), (255, 255, 255), 1)
                    
                    # Draw label text
                    cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw vehicle path if enabled
                    if self.path_tracking_var.get() and tracker_id in self.vehicle_paths:
                        path = self.vehicle_paths[tracker_id]
                        if len(path) > 1:
                            # Draw path with fading effect
                            for j in range(1, len(path)):
                                alpha = j / len(path)  # Fade from transparent to solid
                                path_color = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
                                cv2.line(annotated_frame, 
                                       (int(path[j-1][0]), int(path[j-1][1])), 
                                       (int(path[j][0]), int(path[j][1])), 
                                       path_color, 2)
                    
                    # Draw multi-line status if vehicle has crossed multiple lines
                    if tracker_id in self.vehicle_line_status and len(self.vehicle_line_status[tracker_id]) > 0:
                        lines_crossed = list(self.vehicle_line_status[tracker_id].keys())
                        status_text = f"Lines: {', '.join(lines_crossed)}"
                        (status_width, status_height), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        
                        # Draw status background
                        cv2.rectangle(annotated_frame, (x1, y2), 
                                    (x1 + status_width + 10, y2 + status_height + 10), (0, 0, 0), -1)
                        cv2.rectangle(annotated_frame, (x1, y2), 
                                    (x1 + status_width + 10, y2 + status_height + 10), color, 1)
                        
                        # Draw status text
                        cv2.putText(annotated_frame, status_text, (x1 + 5, y2 + status_height + 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw professional counting lines
        for line in self.counting_lines:
            x1, y1, x2, y2 = line['coords']
            color = line['color']
            
            # Draw line with shadow effect
            cv2.line(annotated_frame, (int(x1)+2, int(y1)+2), (int(x2)+2, int(y2)+2), (0, 0, 0), 4)
            cv2.line(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            
            # Draw professional line label
            label = f" {line['name']} "
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            # Draw label background
            cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_height - 15), 
                        (int(x1) + label_width + 10, int(y1) - 5), color, -1)
            cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_height - 15), 
                        (int(x1) + label_width + 10, int(y1) - 5), (255, 255, 255), 1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (int(x1) + 5, int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add statistics overlay
        self.add_statistics_overlay(annotated_frame)
        
        return annotated_frame
    
    def add_statistics_overlay(self, frame):
        """Add statistics overlay to the frame"""
        # Background for statistics
        overlay_height = 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (10, 10), (400, overlay_height), (255, 255, 255), 2)
        
        # Add statistics text
        y_offset = 35
        cv2.putText(frame, "Vehicle Tracking Statistics", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        # Total vehicles tracked
        cv2.putText(frame, f"Vehicles Tracked: {len(self.vehicle_positions)}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 20
        
        # Frame count
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 20
        
        # CSV logging status
        csv_status = "ACTIVE" if self.csv_logging_var.get() else "INACTIVE"
        cv2.putText(frame, f"CSV Logging: {csv_status}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if csv_status == "ACTIVE" else (0, 0, 255), 1)
        y_offset += 20
        
        # Line counts
        for i, (line_name, counts) in enumerate(list(self.counts.items())[:3]):  # Show first 3 lines
            cv2.putText(frame, f"{line_name}: {counts['total']}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15
    
    def get_previous_line_crossed(self, tracker_id, current_line):
        """Get the previous line crossed by the vehicle"""
        if tracker_id in self.vehicle_history:
            # Look for the most recent crossing before current line
            for crossing in reversed(self.vehicle_history[tracker_id]):
                if crossing['line_name'] != current_line:
                    return crossing['line_name']
        return None
    
    def get_next_line_crossed(self, tracker_id, current_line):
        """Get the next line that the vehicle might cross"""
        # This is a simplified approach - in a real scenario, you might want to
        # predict based on vehicle direction and available lines
        crossed_lines = set()
        if tracker_id in self.vehicle_line_status:
            crossed_lines = set(self.vehicle_line_status[tracker_id].keys())
        
        # Return the next line in the sequence (if any)
        available_lines = [line['name'] for line in self.counting_lines if line['name'] not in crossed_lines]
        return available_lines[0] if available_lines else None
    
    def line_crossed(self, x1, y1, x2, y2, line_x1, line_y1, line_x2, line_y2):
        """Check if line segment (x1,y1)-(x2,y2) crosses line (line_x1,line_y1)-(line_x2,line_y2)"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        A = (x1, y1)
        B = (x2, y2)
        C = (line_x1, line_y1)
        D = (line_x2, line_y2)
        
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    def update_display(self, frame, detections):
        if frame is None:
            return
        
        # Draw vehicle detections with professional bounding boxes
        annotated_frame = frame.copy()
        
        # Draw bounding boxes and vehicle information
        if detections is not None and len(detections) > 0:
            for i, tracker_id in enumerate(detections.tracker_id):
                if tracker_id is not None:
                    bbox = detections.xyxy[i]
                    class_id = int(detections.class_id[i]) if detections.class_id is not None else 2
                    confidence = detections.confidence[i] if detections.confidence is not None else 0.0
                    
                    # Get vehicle type
                    vehicle_type = self.vehicle_types.get(tracker_id, 'car')
                    
                    # Professional color scheme
                    type_colors = {
                        'car': (0, 150, 255),      # Orange
                        'motorcycle': (255, 50, 50),   # Red
                        'bus': (255, 0, 255),      # Magenta
                        'truck': (0, 255, 255),    # Cyan
                        'moped': (255, 255, 0)     # Yellow
                    }
                    color = type_colors.get(vehicle_type, (0, 150, 255))
                    
                    # Draw professional bounding box
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Draw filled rectangle with transparency effect
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
                    
                    # Draw border
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw professional label with background
                    label = f"{tracker_id} | {vehicle_type.upper()} | {confidence:.1f}"
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Draw label background
                    cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), 
                                (x1 + label_width + 10, y1), color, -1)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), 
                                (x1 + label_width + 10, y1), (255, 255, 255), 1)
                    
                    # Draw label text
                    cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw vehicle path if enabled
                    if self.path_tracking_var.get() and tracker_id in self.vehicle_paths:
                        path = self.vehicle_paths[tracker_id]
                        if len(path) > 1:
                            # Draw path with fading effect
                            for j in range(1, len(path)):
                                alpha = j / len(path)  # Fade from transparent to solid
                                path_color = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
                                cv2.line(annotated_frame, 
                                       (int(path[j-1][0]), int(path[j-1][1])), 
                                       (int(path[j][0]), int(path[j][1])), 
                                       path_color, 2)
                    
                    # Draw multi-line status if vehicle has crossed multiple lines
                    if tracker_id in self.vehicle_line_status and len(self.vehicle_line_status[tracker_id]) > 0:
                        lines_crossed = list(self.vehicle_line_status[tracker_id].keys())
                        status_text = f"Lines: {', '.join(lines_crossed)}"
                        (status_width, status_height), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        
                        # Draw status background
                        cv2.rectangle(annotated_frame, (x1, y2), 
                                    (x1 + status_width + 10, y2 + status_height + 10), (0, 0, 0), -1)
                        cv2.rectangle(annotated_frame, (x1, y2), 
                                    (x1 + status_width + 10, y2 + status_height + 10), color, 1)
                        
                        # Draw status text
                        cv2.putText(annotated_frame, status_text, (x1 + 5, y2 + status_height + 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw professional counting lines
        for line in self.counting_lines:
            x1, y1, x2, y2 = line['coords']
            color = line['color']
            
            # Draw line with shadow effect
            cv2.line(annotated_frame, (int(x1)+2, int(y1)+2), (int(x2)+2, int(y2)+2), (0, 0, 0), 4)
            cv2.line(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            
            # Draw professional line label
            label = f" {line['name']} "
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            # Draw label background
            cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_height - 15), 
                        (int(x1) + label_width + 10, int(y1) - 5), color, -1)
            cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_height - 15), 
                        (int(x1) + label_width + 10, int(y1) - 5), (255, 255, 255), 1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (int(x1) + 5, int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Display frame
        self.display_frame(annotated_frame)
        
        # Update statistics
        self.update_statistics()
    
    def update_statistics(self):
        # Update Summary tab
        summary_text = "Vehicle Counts by Type:\n\n"
        
        for line_name, counts in self.counts.items():
            summary_text += f"{line_name}:\n"
            summary_text += f"  Cars: {counts['car']}\n"
            summary_text += f"  Motorcycles: {counts['motorcycle']}\n"
            summary_text += f"  Buses: {counts['bus']}\n"
            summary_text += f"  Trucks: {counts['truck']}\n"
            summary_text += f"  Mopeds: {counts['moped']}\n"
            summary_text += f"  TOTAL: {counts['total']}\n\n"
        
        summary_text += f"Total Vehicles Tracked: {len(self.vehicle_positions)}\n"
        summary_text += f"Frame: {self.frame_count}"
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary_text)
        
        # Update Detailed tab
        detailed_text = "Detailed Crossing Data:\n\n"
        
        for line_name, crossings in self.line_crossings.items():
            detailed_text += f"{line_name} Crossings:\n"
            for crossing in crossings[-10:]:  # Show last 10 crossings
                detailed_text += f"  Vehicle {crossing['vehicle_id']} ({crossing['vehicle_type']}) "
                detailed_text += f"{crossing['direction']} at {crossing['timestamp']:.1f}s\n"
            detailed_text += "\n"
        
        self.detailed_text.delete(1.0, tk.END)
        self.detailed_text.insert(1.0, detailed_text)
        
        # Update Analytics tab
        analytics_text = "Vehicle Analytics:\n\n"
        
        # Multi-line crossing analysis
        multi_line_vehicles = {}
        for vehicle_id, history in self.vehicle_history.items():
            if len(history) > 1:
                lines_crossed = set(crossing['line_name'] for crossing in history)
                if len(lines_crossed) > 1:
                    multi_line_vehicles[vehicle_id] = lines_crossed
        
        analytics_text += f"Vehicles crossing multiple lines: {len(multi_line_vehicles)}\n\n"
        
        for vehicle_id, lines in list(multi_line_vehicles.items())[:10]:  # Show first 10
            vehicle_type = self.vehicle_types.get(vehicle_id, 'unknown')
            analytics_text += f"Vehicle {vehicle_id} ({vehicle_type}): {', '.join(lines)}\n"
        
        # Path analysis
        analytics_text += "\nPath Analysis:\n"
        total_paths = len(self.vehicle_paths)
        avg_path_length = sum(len(path) for path in self.vehicle_paths.values()) / max(1, total_paths)
        analytics_text += f"Total vehicles with paths: {total_paths}\n"
        analytics_text += f"Average path length: {avg_path_length:.1f} points\n"
        
        # Line crossing patterns
        analytics_text += "\nLine Crossing Patterns:\n"
        line_patterns = defaultdict(int)
        for vehicle_id, history in self.vehicle_history.items():
            if len(history) >= 2:
                pattern = []
                for crossing in history:
                    pattern.append(f"{crossing['line_name']}({crossing['direction']})")
                pattern_str = " -> ".join(pattern)
                line_patterns[pattern_str] += 1
        
        for pattern, count in sorted(line_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
            analytics_text += f"{pattern}: {count} vehicles\n"
        
        analytics_text += "\nVehicle Type Distribution:\n"
        type_counts = defaultdict(int)
        for vehicle_type in self.vehicle_types.values():
            type_counts[vehicle_type] += 1
        
        for vehicle_type, count in type_counts.items():
            analytics_text += f"{vehicle_type}: {count}\n"
        
        # CSV logging status
        if self.csv_logging_var.get():
            analytics_text += f"\nCSV Logging: ACTIVE\n"
            if self.csv_path:
                analytics_text += f"File: {os.path.basename(self.csv_path)}\n"
        else:
            analytics_text += f"\nCSV Logging: INACTIVE\n"
        
        self.analytics_text.delete(1.0, tk.END)
        self.analytics_text.insert(1.0, analytics_text)
    
    def export_data(self):
        """Export all counting data to JSON and CSV files"""
        if not self.counting_lines:
            messagebox.showwarning("Warning", "No data to export")
            return
        
        # Create output directory
        os.makedirs("./output", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Export JSON data
        json_data = {
            'video_path': self.video_path,
            'processing_time': time.time(),
            'frame_count': self.frame_count,
            'counting_lines': self.counting_lines,
            'vehicle_counts': dict(self.counts),
            'line_crossings': {k: v for k, v in self.line_crossings.items()},
            'vehicle_history': {str(k): v for k, v in self.vehicle_history.items()},
            'vehicle_types': {str(k): v for k, v in self.vehicle_types.items()},
            'settings': {
                'model_type': self.model_var.get(),
                'confidence_threshold': self.conf_threshold_var.get(),
                'skip_frames': self.skip_frames_var.get(),
                'min_vehicle_size': self.min_size_var.get(),
                'processing_speed': self.speed_var.get()
            }
        }
        
        json_path = f"./output/vehicle_data_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        # Export CSV summary
        csv_path = f"./output/vehicle_summary_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(['Line Name', 'Car', 'Motorcycle', 'Bus', 'Truck', 'Moped', 'Total'])
            for line_name, counts in self.counts.items():
                writer.writerow([
                    line_name,
                    counts['car'],
                    counts['motorcycle'],
                    counts['bus'],
                    counts['truck'],
                    counts['moped'],
                    counts['total']
                ])
        
        # Export detailed crossings CSV
        crossings_path = f"./output/crossings_{timestamp}.csv"
        with open(crossings_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Vehicle ID', 'Vehicle Type', 'Line Name', 'Direction', 'From Line', 'To Line', 'Confidence'])
            for line_name, crossings in self.line_crossings.items():
                for crossing in crossings:
                    writer.writerow([
                        crossing['timestamp'],
                        crossing['vehicle_id'],
                        crossing['vehicle_type'],
                        crossing['line_name'],
                        crossing['direction'],
                        crossing.get('from_line', ''),
                        crossing.get('to_line', ''),
                        crossing['confidence']
                    ])
        
        # Export vehicle paths CSV
        paths_path = f"./output/vehicle_paths_{timestamp}.csv"
        with open(paths_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Vehicle ID', 'Vehicle Type', 'Path Point', 'X', 'Y', 'Timestamp'])
            for vehicle_id, path in self.vehicle_paths.items():
                vehicle_type = self.vehicle_types.get(vehicle_id, 'unknown')
                for i, (x, y) in enumerate(path):
                    writer.writerow([vehicle_id, vehicle_type, i, x, y, time.time()])
        
        # Export multi-line analysis CSV
        multi_line_path = f"./output/multi_line_analysis_{timestamp}.csv"
        with open(multi_line_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Vehicle ID', 'Vehicle Type', 'Lines Crossed', 'Crossing Pattern', 'Total Distance', 'Path Length'])
            for vehicle_id, history in self.vehicle_history.items():
                if len(history) > 1:
                    vehicle_type = self.vehicle_types.get(vehicle_id, 'unknown')
                    lines_crossed = set(crossing['line_name'] for crossing in history)
                    pattern = " -> ".join([f"{crossing['line_name']}({crossing['direction']})" for crossing in history])
                    
                    # Calculate total distance
                    total_distance = 0
                    if vehicle_id in self.vehicle_paths and len(self.vehicle_paths[vehicle_id]) > 1:
                        for i in range(1, len(self.vehicle_paths[vehicle_id])):
                            prev = self.vehicle_paths[vehicle_id][i-1]
                            curr = self.vehicle_paths[vehicle_id][i]
                            total_distance += np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
                    
                    path_length = len(self.vehicle_paths.get(vehicle_id, []))
                    writer.writerow([vehicle_id, vehicle_type, len(lines_crossed), pattern, f"{total_distance:.1f}", path_length])
        
        messagebox.showinfo("Success", f"Data exported successfully!\n\nJSON: {json_path}\nSummary CSV: {csv_path}\nCrossings CSV: {crossings_path}\nPaths CSV: {paths_path}\nMulti-line Analysis: {multi_line_path}")
    
    def save_processed_video(self):
        """Save the current processed video"""
        if not self.output_path or not os.path.exists(self.output_path):
            messagebox.showwarning("Warning", "No processed video available. Please process a video first.")
            return
        
        # Ask user for save location
        save_path = filedialog.asksaveasfilename(
            title="Save Processed Video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
            initialvalue=os.path.basename(self.output_path)
        )
        
        if save_path:
            try:
                import shutil
                shutil.copy2(self.output_path, save_path)
                messagebox.showinfo("Success", f"Processed video saved to: {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save video: {e}")

def main():
    root = tk.Tk()
    app = VehicleCountingGUI(root)
    
    # Add cleanup function for CSV logging
    def on_closing():
        if app.csv_file:
            app.stop_csv_logging()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
