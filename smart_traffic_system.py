import cv2
import numpy as np
import time
import os
from collections import deque
from ultralytics import YOLO
import torch
import random
import glob

class AdvancedTrafficLightSystem:
    def __init__(self):

        self.loaded_images = {}

        # Initialize traffic light states
        self.lights = {'Road_A': 'RED', 'Road_B': 'RED', 'Road_C': 'RED'}
        self.vehicle_counts = {'Road_A': 0, 'Road_B': 0, 'Road_C': 0}
        self.vehicle_types = {'Road_A': {}, 'Road_B': {}, 'Road_C': {}}
        
        # Load YOLOv8 model
        print("🚀 Loading YOLOv8 model for vehicle detection...")
        try:
            self.model = YOLO('yolov8n.pt')
            print("✅ YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            return
        
        # Vehicle classes in COCO dataset
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # Enhanced timing parameters
        self.min_green_time = 14
        self.max_green_time = 22
        self.yellow_time = 3
        self.current_green = None
        self.green_start_time = 0
        self.yellow_start_time = 0
        self.is_yellow_phase = False
        self.yellow_previous_road = None
        
        # Traffic history for smart decisions
        self.traffic_history = {
            'Road_A': deque(maxlen=10),
            'Road_B': deque(maxlen=10), 
            'Road_C': deque(maxlen=10)
        }
        
        self.waiting_times = {'Road_A': 0, 'Road_B': 0, 'Road_C': 0}
        self.road_cycles = {'Road_A': 0, 'Road_B': 0, 'Road_C': 0}
        
        # Runtime control
        self.start_time = None
        self.total_runtime = 300
        
        # Performance tracking
        self.performance_stats = {
            'total_detections': 0,
            'frame_count': 0,
            'light_changes': 0
        }

    def setup_media_sources(self):
        """Support multiple folders with videos and random image selection"""
        
        # Define your folder structure
        media_folders = {
            'video_set_1': 'videos_set_1',
            'video_set_2': 'videos_set_2', 
            'images': 'images_folder'
        }
        
        # Check which folders exist
        available_folders = {}
        for set_name, folder_path in media_folders.items():
            if os.path.exists(folder_path):
                available_folders[set_name] = folder_path
                print(f"✅ Found {set_name}: {folder_path}")
            else:
                print(f"⚠️  Missing {set_name}: {folder_path}")
        
        if not available_folders:
            print("❌ No media folders found! Using current directory as fallback...")
            return self.fallback_media_setup()
        
        # Let user choose which set to use
        print("\n📁 Available media sets:")
        for i, set_name in enumerate(available_folders.keys(), 1):
            print(f"   {i}. {set_name}")
        
        try:
            choice = input("🎯 Choose media set (number) or press Enter for auto-selection: ")
            if choice.strip() == '':
                # Auto-select: prefer video sets over images
                if 'video_set_1' in available_folders:
                    selected_set = 'video_set_1'
                elif 'video_set_2' in available_folders:
                    selected_set = 'video_set_2'
                else:
                    selected_set = 'images'
            else:
                set_names = list(available_folders.keys())
                selected_set = set_names[int(choice) - 1]
            
            print(f"🎬 Selected: {selected_set}")
            return self.load_selected_media(available_folders[selected_set], selected_set)
            
        except (ValueError, IndexError):
            print("⚠️  Invalid choice, using auto-selection")
            return self.fallback_media_setup()

    def load_selected_media(self, folder_path, set_type):
        """Load media files from selected folder - FIXED VERSION"""
        road_names = ['Road_A', 'Road_B', 'Road_C']
        media_files = []
        
        if set_type.startswith('video'):
            print(f"📁 Scanning folder: {folder_path}")
            
            # Get all video files in the folder
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
            all_videos = []
            
            for ext in video_extensions:
                all_videos.extend(glob.glob(os.path.join(folder_path, ext)))
                all_videos.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
            print(f"🎬 Found {len(all_videos)} video files: {[os.path.basename(v) for v in all_videos]}")
            
            if len(all_videos) < 3:
                print(f"❌ Need at least 3 videos in {folder_path}, found {len(all_videos)}")
                return None, None
            
            # Assign videos to roads - try to match patterns first, then use any 3 videos
            assigned_videos = []
            
            # Try to find specific road patterns
            road_patterns = {
                'Road_A': ['road_a', 'road_A', 'a', 'A', '1', 'one'],
                'Road_B': ['road_b', 'road_B', 'b', 'B', '2', 'two'], 
                'Road_C': ['road_c', 'road_C', 'c', 'C', '3', 'three']
            }
            
            for road in road_names:
                road_assigned = False
                for video_path in all_videos:
                    if video_path not in assigned_videos:
                        video_name = os.path.basename(video_path).lower()
                        # Check if video name contains any pattern for this road
                        for pattern in road_patterns[road]:
                            if pattern.lower() in video_name:
                                media_files.append(('video', video_path))
                                assigned_videos.append(video_path)
                                print(f"✅ {road}: {os.path.basename(video_path)} (pattern matched)")
                                road_assigned = True
                                break
                        if road_assigned:
                            break
                
                # If no pattern match, assign any remaining video
                if not road_assigned:
                    for video_path in all_videos:
                        if video_path not in assigned_videos:
                            media_files.append(('video', video_path))
                            assigned_videos.append(video_path)
                            print(f"✅ {road}: {os.path.basename(video_path)} (auto-assigned)")
                            break
            
        else:  # images
            # For images, randomly select 3 images
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            all_images = []
            
            for ext in image_extensions:
                all_images.extend(glob.glob(os.path.join(folder_path, ext)))
                all_images.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
            print(f"🖼️ Found {len(all_images)} image files")
            
            if len(all_images) < 3:
                print(f"❌ Need at least 3 images in {folder_path}, found {len(all_images)}")
                return None, None
            
            # Randomly select 3 unique images
            selected_images = random.sample(all_images, 3)
            
            for i, img_path in enumerate(selected_images):
                road = road_names[i]
                media_files.append(('image', img_path))
                print(f"✅ {road}: {os.path.basename(img_path)}")
        
        return media_files, road_names

    def fallback_media_setup(self):
        """Fallback to current directory files"""
        print("🔄 Using fallback media setup (current directory)")
        
        road_names = ['Road_A', 'Road_B', 'Road_C']
        media_files = []
        
        for road in road_names:
            # Try various file patterns in current directory
            possible_files = [
                f'{road.lower()}.mp4', f'{road.upper()}.MP4',
                f'{road.lower()}.avi', f'{road.upper()}.AVI',
                f'{road.lower()}.jpg', f'{road.upper()}.JPG',
                f'{road.lower()}.png', f'{road.upper()}.PNG'
            ]
            
            found = False
            for file in possible_files:
                if os.path.exists(file):
                    media_type = 'video' if file.lower().endswith(('.mp4', '.avi')) else 'image'
                    media_files.append((media_type, file))
                    print(f"✅ {road}: {file}")
                    found = True
                    break
            
            if not found:
                print(f"❌ No media file found for {road}")
                return None, None
        
        return media_files, road_names

    def detect_vehicles_fixed(self, frame, road_name):
        """
        FIXED vehicle detection with proper frame handling
        """
        try:
            # Make a copy for display
            display_frame = frame.copy()
            
            # Resize for consistent processing
            height, width = frame.shape[:2]
            if width > 1280:
                scale = 1280 / width
                new_width = 1280
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Run YOLOv8 inference
            results = self.model(frame, conf=0.5, verbose=False)
            
            vehicle_count = 0
            vehicle_types = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = box.conf[0]
                        
                        # Filter for vehicles with good confidence
                        if class_id in self.vehicle_classes and confidence > 0.5:
                            vehicle_count += 1
                            
                            # Count vehicle types
                            if class_id == 2: 
                                vehicle_types['car'] += 1
                            elif class_id == 3: 
                                vehicle_types['motorcycle'] += 1
                            elif class_id == 5: 
                                vehicle_types['bus'] += 1
                            elif class_id == 7: 
                                vehicle_types['truck'] += 1
                            
                            # Draw bounding box on the display frame
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Scale coordinates back to original frame if needed
                            if width > 1280:
                                x1 = int(x1 / scale)
                                y1 = int(y1 / scale)
                                x2 = int(x2 / scale)
                                y2 = int(y2 / scale)
                            
                            label = f"{self.model.names[class_id]} {confidence:.2f}"
                            
                            color_map = {
                                2: (0, 255, 0),    # Green for cars
                                3: (255, 255, 0),  # Cyan for motorcycles
                                5: (0, 255, 255),  # Yellow for buses
                                7: (255, 0, 0)     # Red for trucks
                            }
                            color = color_map.get(class_id, (0, 255, 0))
                            
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(display_frame, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            self.performance_stats['total_detections'] += vehicle_count
            self.performance_stats['frame_count'] += 1
            
            # Debug output occasionally
            if self.performance_stats['frame_count'] % 50 == 0:
                print(f"🔍 {road_name}: Detected {vehicle_count} vehicles")
            
            return vehicle_count, display_frame, vehicle_types
            
        except Exception as e:
            print(f"❌ Detection error in {road_name}: {e}")
            return 0, frame, {}

    def calculate_traffic_priority(self, road_name):
        """
        FIXED priority calculation without crazy cycle counts
        """
        current_count = self.vehicle_counts[road_name]
        
        # Reasonable waiting time emphasis
        waiting_factor = min(self.waiting_times[road_name] / 20.0, 3.0)
        
        # Small bonus for roads that haven't gotten green recently
        cycle_bonus = min(self.road_cycles[road_name] * 0.5, 2.0)
        
        # Vehicle type weighting
        type_weights = {'car': 1.0, 'motorcycle': 0.6, 'bus': 3.5, 'truck': 2.8}
        weighted_count = 0
        for vehicle_type, count in self.vehicle_types[road_name].items():
            weighted_count += count * type_weights.get(vehicle_type, 1.0)
        
        # BALANCED PRIORITY FORMULA
        priority = (
            0.50 * weighted_count +           # 50% for current traffic
            0.35 * waiting_factor +           # 35% for waiting time
            0.15 * cycle_bonus                # 15% for cycle fairness
        )
        
        return max(priority, 0)

    def update_road_cycles(self):
        """FIXED cycle counting"""
        for road in self.road_cycles:
            if road != self.current_green:
                self.road_cycles[road] += 1
            else:
                self.road_cycles[road] = 0  # Reset when road gets green

    def update_waiting_times(self):
        """FIXED waiting time updates"""
        for road in self.waiting_times:
            if road != self.current_green or self.lights[road] != 'GREEN':
                self.waiting_times[road] += 1
            else:
                self.waiting_times[road] = 0  # Reset when green

    def calculate_green_time(self):
        """Calculate reasonable green time"""
        if self.current_green is None:
            return self.min_green_time
            
        current_count = self.vehicle_counts[self.current_green]
        
        if current_count == 0:
            return self.min_green_time
        
        # Reasonable time per vehicle
        base_time_per_vehicle = 2.0
        
        # Vehicle type adjustments
        type_adjustment = 0
        for vehicle_type, count in self.vehicle_types[self.current_green].items():
            if vehicle_type == 'bus':
                type_adjustment += count * 1.5
            elif vehicle_type == 'truck':
                type_adjustment += count * 1.0
        
        clearance_time = 3.0
        
        green_time = (
            self.min_green_time +
            (current_count * base_time_per_vehicle) +
            type_adjustment +
            clearance_time
        )
        
        return min(green_time, self.max_green_time)

    def update_lights_intelligent(self):
        """
        FIXED intelligent traffic light control
        """
        current_time = time.time()
        
        # Handle yellow phase first
        if self.is_yellow_phase:
            yellow_elapsed = current_time - self.yellow_start_time
            if yellow_elapsed >= self.yellow_time:
                # Yellow phase complete, switch to new green
                self.lights[self.yellow_previous_road] = 'RED'
                self.lights[self.current_green] = 'GREEN'
                self.green_start_time = current_time
                self.is_yellow_phase = False
                self.waiting_times[self.current_green] = 0
                print(f"🟢 NEW: {self.current_green} gets GREEN for {self.calculate_green_time():.1f}s "
                      f"(Vehicles: {self.vehicle_counts[self.current_green]})")
            return 1  # Check again soon during yellow phase
        
        # Initialize system
        if self.current_green is None:
            self.current_green = max(self.vehicle_counts, key=self.vehicle_counts.get)
            self.lights[self.current_green] = 'GREEN'
            self.green_start_time = current_time
            self.waiting_times[self.current_green] = 0
            initial_time = self.calculate_green_time()
            print(f"🚦 SYSTEM START: {self.current_green} gets GREEN for {initial_time:.1f}s")
            return initial_time
        
        self.update_waiting_times()
        self.update_road_cycles()
        
        elapsed_green = current_time - self.green_start_time
        
        # MINIMUM GUARANTEED GREEN TIME
        if elapsed_green < self.min_green_time:
            return self.min_green_time - elapsed_green
        
        # Calculate priorities
        priorities = {}
        for road in ['Road_A', 'Road_B', 'Road_C']:
            priorities[road] = self.calculate_traffic_priority(road)
        
        # Get allowed next roads (collision prevention)
        if self.current_green == 'Road_A':
            allowed_next = ['Road_C', 'Road_B']
        elif self.current_green == 'Road_B':
            allowed_next = ['Road_C', 'Road_A']
        else:  # Road_C
            allowed_next = ['Road_A', 'Road_B']
        
        # Find best candidate
        candidate_priority = -1
        candidate_road = None
        
        for road in allowed_next:
            if priorities[road] > candidate_priority:
                candidate_priority = priorities[road]
                candidate_road = road
        
        current_priority = priorities[self.current_green]
        
        # REASONABLE SWITCHING CONDITIONS
        should_switch = False
        switch_reason = ""
        
        # Condition 1: Significant priority advantage
        if candidate_priority > current_priority + 1.5:
            should_switch = True
            switch_reason = f"Priority advantage ({candidate_priority:.1f} vs {current_priority:.1f})"
        
        # Condition 2: Maximum green time reached
        elif elapsed_green >= self.max_green_time:
            should_switch = True
            switch_reason = f"Max time ({self.max_green_time}s)"
        
        # Condition 3: Emergency waiting situation
        elif max(self.waiting_times.values()) > 30:
            longest_wait = max(self.waiting_times, key=self.waiting_times.get)
            if longest_wait in allowed_next:
                candidate_road = longest_wait
                should_switch = True
                switch_reason = f"Emergency wait ({self.waiting_times[longest_wait]}s)"
        
        # Condition 4: Force rotation if any road severely neglected
        elif max(self.road_cycles.values()) > 5 and elapsed_green >= 15:
            most_neglected = max(self.road_cycles, key=self.road_cycles.get)
            if most_neglected in allowed_next:
                candidate_road = most_neglected
                should_switch = True
                switch_reason = f"Fairness rotation (cycle {self.road_cycles[most_neglected]})"
        
        # Execute switch if conditions met
        if should_switch and candidate_road and candidate_road != self.current_green:
            self.performance_stats['light_changes'] += 1
            
            print(f"🟡 SWITCHING: {self.current_green} → {candidate_road} | Reason: {switch_reason}")
            
            # Start yellow phase
            self.is_yellow_phase = True
            self.yellow_previous_road = self.current_green
            self.lights[self.current_green] = 'YELLOW'
            self.yellow_start_time = current_time
            self.current_green = candidate_road
            
            return 1  # Short return during yellow phase
        
        return min(self.calculate_green_time() - elapsed_green, 2)  # Check every 2 seconds max
    
    def get_media_frame(self, media_type, cap, file_path, frame_count):
        """Get frame from video or image with looping support"""
        if media_type == 'video':
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    # Loop video - reset to beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if ret:
                        print(f"🔄 Video looped: {os.path.basename(file_path)}")
                return ret, frame
            return False, None
        else:  # image
            # For images, load once and reuse
            if file_path not in self.loaded_images:
                self.loaded_images[file_path] = cv2.imread(file_path)
                if self.loaded_images[file_path] is None:
                    print(f"❌ Failed to load image: {file_path}")
                    return False, None
            
            # Optional: Add slight variations to static images to make detection more interesting
            img = self.loaded_images[file_path].copy()
            
            # Add timestamp text to make each frame slightly different
            cv2.putText(img, f"Frame: {frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return True, img

    def get_remaining_time(self):
        if self.start_time is None:
            return self.total_runtime
        elapsed = time.time() - self.start_time
        return max(0, self.total_runtime - elapsed)

    def display_enhanced_info(self, frame, road_name):
        """Display comprehensive information"""
        color = self.get_light_color(self.lights[road_name])
        priority = self.calculate_traffic_priority(road_name)
        
        # Main information
        cv2.putText(frame, f"{road_name} - Smart Traffic System", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Vehicles: {self.vehicle_counts[road_name]}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Light status with background
        light_text = f"LIGHT: {self.lights[road_name]}"
        text_size = cv2.getTextSize(light_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (10, 10), (20 + text_size[0], 45), color, -1)
        cv2.putText(frame, light_text, (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Traffic analytics
        y_offset = 90
        info_lines = [
            f"Priority: {priority:.1f}",
            f"Waiting: {self.waiting_times[road_name]}s",
            f"Cycle: {self.road_cycles[road_name]}",
        ]
        
        for line in info_lines:
            cv2.putText(frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # Vehicle type breakdown
        if any(count > 0 for count in self.vehicle_types[road_name].values()):
            cv2.putText(frame, "Vehicle Types:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
            for v_type, count in self.vehicle_types[road_name].items():
                if count > 0:
                    cv2.putText(frame, f"  {v_type}: {count}", (15, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
                    y_offset += 15

    def arrange_windows(self):
        """Enhanced window arrangement with better positioning"""
        try:
            # Set window properties first
            for road in ['Road_A', 'Road_B', 'Road_C']:
                cv2.namedWindow(road, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(road, 640, 480)  # Standard size
            
            # Position windows in a clean layout
            cv2.moveWindow('Road_A', 100, 100)    # Top-left
            cv2.moveWindow('Road_B', 800, 100)    # Top-right  
            cv2.moveWindow('Road_C', 450, 600)    # Bottom-center
            
            print("✅ Windows auto-arranged in grid layout")
        except Exception as e:
            print(f"⚠️  Window arrangement warning: {e}")
            
    def run_complete_system(self):
        """Main function with flexible media support"""
        # Setup media sources using the new folder system
        media_info, road_names = self.setup_media_sources()
        if not media_info:
            print("❌ No valid media files found!")
            print("💡 Please create folders: videos_set_1/, videos_set_2/, or images_folder/")
            print("💡 Or place media files in current directory: road_A.mp4, road_B.mp4, road_C.mp4")
            return
        
        # Create media captures
        media_captures = []
        for media_type, file_path in media_info:
            if media_type == 'video':
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    print(f"❌ Cannot open video: {file_path}")
                    return
            else:
                cap = None  # Images don't need VideoCapture
            media_captures.append((media_type, cap, file_path))
        
        # Auto-arrange windows
        self.arrange_windows()
        
        # FIXED RUNTIME: Use fixed time instead of video length (since videos loop)
        self.total_runtime = 300  # 5 minutes fixed runtime
        print(f"⏱️  System will run for {self.total_runtime} seconds (videos will loop)")
        
        # Show video info but don't use for runtime
        for media_type, cap, file_path in media_captures:
            if media_type == 'video' and cap is not None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps if fps > 0 else 0
                print(f"📹 {os.path.basename(file_path)}: {duration:.1f} seconds (will loop)")
            else:  # image
                print(f"🖼️ {os.path.basename(file_path)}: Static image")
        
        self.start_time = time.time()
        frame_count = 0
        last_status_time = 0
        
        try:
            while self.get_remaining_time() > 0:
                frame_count += 1
                current_time = time.time()
                remaining = self.get_remaining_time()
                
                # Process all media feeds
                for i, (media_type, cap, file_path) in enumerate(media_captures):
                    road = road_names[i]
                    
                    ret, frame = self.get_media_frame(media_type, cap, file_path, frame_count)
                    
                    if ret and frame is not None:
                        count, processed_frame, vehicle_types = self.detect_vehicles_fixed(frame, road)
                        self.vehicle_counts[road] = count
                        self.vehicle_types[road] = vehicle_types
                        self.traffic_history[road].append(count)
                        
                        self.display_enhanced_info(processed_frame, road)
                        
                        # Add media type and time info
                        media_indicator = "📹" if media_type == 'video' else "🖼️"
                        cv2.putText(processed_frame, f"Time: {remaining:.0f}s", 
                                (10, processed_frame.shape[0] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(processed_frame, f"{media_indicator} {media_type}", 
                                (processed_frame.shape[1] - 120, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        cv2.imshow(road, processed_frame)
                
                # Update traffic lights
                next_check = self.update_lights_intelligent()
                
                # Display status every 5 seconds
                if current_time - last_status_time >= 5.0:
                    self.display_system_status(remaining)
                    last_status_time = current_time
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or remaining <= 0:
                    break
                    
                time.sleep(0.03)
                
        except KeyboardInterrupt:
            print("\n⏹️  System stopped by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.display_final_report()
            # Cleanup - only release video captures
            for media_type, cap, file_path in media_captures:
                if media_type == 'video' and cap is not None:
                    cap.release()
            cv2.destroyAllWindows()

    def display_system_status(self, remaining):
        """Display system status"""
        status = f"⏱️ [{remaining:4.0f}s] "
        
        for road in ['Road_A', 'Road_B', 'Road_C']:
            light_icon = {'RED': '🔴', 'YELLOW': '🟡', 'GREEN': '🟢'}[self.lights[road]]
            status += f"{light_icon}{road}:{self.vehicle_counts[road]:2d} "
        
        print(status)
        
        detail = "   📊 "
        for road in ['Road_A', 'Road_B', 'Road_C']:
            detail += f"{road}(P:{self.calculate_traffic_priority(road):.1f}, W:{self.waiting_times[road]}s) "
        
        print(detail)

    def display_final_report(self):
        """Display final report"""
        print("\n" + "=" * 80)
        print("🏁 FINAL TRAFFIC LIGHT SYSTEM REPORT")
        print("=" * 80)
        
        print("📊 TRAFFIC STATISTICS:")
        for road in ['Road_A', 'Road_B', 'Road_C']:
            avg_traffic = np.mean(list(self.traffic_history[road])) if self.traffic_history[road] else 0
            type_summary = ", ".join([f"{k}:{v}" for k, v in self.vehicle_types[road].items() if v > 0])
            if not type_summary:
                type_summary = "none"
            print(f"   {road}: {self.vehicle_counts[road]} vehicles (avg: {avg_traffic:.1f})")
            print(f"     Types: {type_summary} | Light: {self.lights[road]} | Final wait: {self.waiting_times[road]}s")
        
        print(f"\n⚡ PERFORMANCE: {self.performance_stats['light_changes']} light changes")
        print(f"📈 Total detections: {self.performance_stats['total_detections']}")

    def get_light_color(self, state):
        colors = {'RED': (0, 0, 255), 'YELLOW': (0, 255, 255), 'GREEN': (0, 255, 0)}
        return colors.get(state, (255, 255, 255))

# GPU Debugging
def check_gpu_status():
    print("\n🔍 GPU STATUS CHECK:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print("✅ GPU acceleration enabled! 🚀")
    else:
        print("❌ GPU not available. Using CPU mode.")
        print("💡 For better performance, install CUDA-compatible PyTorch")

if __name__ == "__main__":
    check_gpu_status()
    print("\n" + "=" * 80)
    system = AdvancedTrafficLightSystem()
    system.run_complete_system()