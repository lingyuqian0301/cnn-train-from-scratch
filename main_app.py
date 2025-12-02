"""
Main Integration App & Configuration
Integrates inference and sorting, manages UI and logging
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import csv
import cv2
import numpy as np
import time
import torch

# Import from object-detection.py
# Note: Python module names use underscores, so we import from the file directly
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("object_detection", "object-detection.py")
object_detection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(object_detection)
WebcamPredictor = object_detection.WebcamPredictor

# Import sorting simulator
from sorting_simulator import SortingSimulator, SortingInstruction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    """Application configuration"""
    model_path: str = "saved_models/best_model.pth"
    confidence_threshold: float = 0.5
    fps_target: int = 20
    enable_webcam: bool = True
    enable_sorting: bool = True
    enable_logging: bool = True
    log_file: str = "logs/classifications.csv"
    sorting_log_file: str = "logs/sorting_log.json"
    display_fps: bool = True
    display_confidence: bool = True
    roi_top_margin: int = 300  # Pixels to exclude from top for inference
    roi_bottom_margin: int = 150  # Pixels to exclude from bottom for inference
    show_roi_indicator: bool = True  # Show rectangle indicating detection area
    debug_show_roi_window: bool = False  # Show cropped ROI in separate preview window
    highlight_duration: float = 1.5  # Seconds to keep detection highlight active
    enable_object_detection: bool = True
    object_min_area: int = 3000
    object_max_area: int = 250000
    max_objects_per_frame: int = 12
    bbox_padding: int = 6
    
    @classmethod
    def load_from_file(cls, config_path: str = "config.json") -> 'AppConfig':
        """Load configuration from JSON file"""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        else:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return cls()
    
    def save_to_file(self, config_path: str = "config.json"):
        """Save configuration to JSON file"""
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Configuration saved to {config_path}")

class ClassificationLogger:
    """Logs all classifications to CSV"""
    
    def __init__(self, log_file: str = "logs/classifications.csv"):
        """Initialize logger"""
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create CSV header if file doesn't exist
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'predicted_class', 'confidence', 
                    'bin_location', 'verification_status', 'manual_review'
                ])
    
    def log_classification(self, predicted_class: str, confidence: float,
                          bin_location: Optional[str] = None,
                          verification_status: Optional[str] = None,
                          manual_review: bool = False):
        """Log a classification to CSV"""
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    predicted_class,
                    f"{confidence:.4f}",
                    bin_location or "N/A",
                    verification_status or "N/A",
                    str(manual_review)
                ])
        except Exception as e:
            logger.error(f"Failed to log classification: {e}")

class DashboardRenderer:
    """Renders the real-time dashboard on video frames"""
    
    def __init__(self, config: AppConfig):
        """Initialize dashboard renderer"""
        self.config = config
    
    def render_inference_info(self, frame: np.ndarray, predicted_class: str,
                             confidence: float, verification_status: str,
                             status_color: tuple) -> np.ndarray:
        """Add inference information to frame"""
        h, w = frame.shape[:2]
        annotated = frame.copy()
        
        # Title background
        cv2.rectangle(annotated, (0, 0), (w, 60), (0, 0, 0), -1)
        
        # Title
        cv2.putText(annotated, "WASTE CLASSIFICATION SYSTEM", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Prediction box
        text = f"{predicted_class.upper()} - {confidence:.1%}"
        color = (0, 255, 0) if confidence >= 0.75 else (0, 165, 255) if confidence >= 0.5 else (0, 0, 255)
        
        cv2.rectangle(annotated, (20, 80), (w - 20, 130), color, 2)
        cv2.putText(annotated, text, (30, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        
        # Verification status
        cv2.putText(annotated, verification_status, (30, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return annotated
    
    def render_sorting_info(self, frame: np.ndarray, instruction: str,
                           bin_location: str, bin_color: tuple) -> np.ndarray:
        """Add sorting information to frame"""
        h, w = frame.shape[:2]
        annotated = frame.copy()
        
        # Sorting section background
        cv2.rectangle(annotated, (20, h - 100), (w - 20, h - 20), 
                     (0, 0, 0), -1)
        
        # Instruction text
        cv2.putText(annotated, "SORTING INSTRUCTION:", (30, h - 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.putText(annotated, bin_location, (30, h - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, bin_color, 2)
        
        return annotated
    
    def render_statistics(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        """Add statistics overlay to frame"""
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Stats background (top right)
        stats_x = w - 320
        stats_y = 150
        stats_w = 300
        stats_h = 200
        
        cv2.rectangle(annotated, (stats_x, stats_y), (stats_x + stats_w, stats_y + stats_h), (0, 0, 0), -1)
        cv2.rectangle(annotated, (stats_x, stats_y), (stats_x + stats_w, stats_y + stats_h), (200, 200, 200), 2)
        
        # Stats title
        cv2.putText(annotated, "STATISTICS", (stats_x + 10, stats_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        y_offset = stats_y + 50
        stats_lines = [
            f"Total Sorted: {stats.get('total_items_sorted', 0)}",
            f"Auto Classified: {stats.get('high_confidence_sorts', 0)}",
            f"Needs Verify: {stats.get('medium_confidence_sorts', 0)}",
            f"Manual Review: {stats.get('manual_reviews_required', 0)}",
            f"Total Instructions: {stats.get('total_instructions', 0)}",
        ]
        
        for line in stats_lines:
            cv2.putText(annotated, line, (stats_x + 15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
        
        return annotated
    
    def render_keyboard_controls(self, frame: np.ndarray) -> np.ndarray:
        """Add keyboard controls legend to frame"""
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Controls background (bottom right)
        controls_x = w - 250
        controls_y = h - 120
        controls_w = 230
        controls_h = 100
        
        cv2.rectangle(annotated, (controls_x, controls_y), (controls_x + controls_w, controls_y + controls_h), 
                     (0, 0, 0), -1)
        cv2.rectangle(annotated, (controls_x, controls_y), (controls_x + controls_w, controls_y + controls_h), 
                     (100, 100, 255), 2)
        
        # Controls title
        cv2.putText(annotated, "KEYBOARD CONTROLS", (controls_x + 10, controls_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        
        y_offset = controls_y + 40
        controls = [
            ("'q'", "Quit"),
            ("'s'", "Toggle Stats"),
            ("'r'", "Reset Stats"),
            ("'e'", "Empty Bins"),
        ]
        
        for key, desc in controls:
            text = f"{key}: {desc}"
            cv2.putText(annotated, text, (controls_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 20
        
        return annotated

class RealTimeClassificationApp:
    """Main application coordinating all components"""
    
    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize application"""
        self.config = config or AppConfig.load_from_file()
        self.logger = ClassificationLogger(self.config.log_file)
        self.dashboard = DashboardRenderer(self.config)
        
        # Statistics tracking
        self.stats = {
            'total_items_sorted': 0,
            'high_confidence_sorts': 0,
            'medium_confidence_sorts': 0,
            'manual_reviews_required': 0,
            'total_instructions': 0
        }
        
        self.show_stats = True
        
        # Initialize inference engine
        logger.info(f"Loading model from {self.config.model_path}")
        self.inference_engine = WebcamPredictor(self.config.model_path)
        
        # Initialize sorting simulator with JSON logging
        self.sorting_simulator = SortingSimulator(
            max_bin_capacity=100,
            log_file=self.config.sorting_log_file
        )
        
        logger.info("Application initialized")
    
    def update_statistics(self, confidence: float, has_instruction: bool):
        """Update statistics based on classification"""
        if confidence >= 0.75:
            self.stats['high_confidence_sorts'] += 1
            self.stats['total_items_sorted'] += 1
        elif confidence >= 0.5:
            self.stats['medium_confidence_sorts'] += 1
            self.stats['total_items_sorted'] += 1
        else:
            self.stats['manual_reviews_required'] += 1
        
        if has_instruction:
            self.stats['total_instructions'] += 1
    
    def run(self):
        """Run the application"""
        if not self.config.enable_webcam:
            logger.error("Webcam is disabled in configuration")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Error: Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        logger.info("Webcam started. Press 'q' to quit, 's' to toggle stats, 'r' to reset stats.")
        
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        last_log_time = time.time()
        log_interval = 1.0  # Log every 1 second to avoid spam
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Error: Could not read frame")
                break
            
            fps_frame_count += 1
            if fps_frame_count >= 30:
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
            
            # Get prediction
            predicted_class, confidence = self.inference_engine.predict_frame(frame)
            
            if predicted_class is not None:
                # Generate sorting instruction using the new simulator
                instruction: SortingInstruction = self.sorting_simulator.generate_instruction(
                    predicted_class, confidence
                )
                
                sorting_instruction = instruction.instruction_text
                bin_location = instruction.bin_location
                has_instruction = not instruction.requires_manual_review
                
                # Get verification status and bin color from sorting simulator
                verification_status, status_color = self.sorting_simulator.get_verification_status(confidence)
                bin_color = self.sorting_simulator.get_bin_color(predicted_class) if has_instruction else (128, 128, 128)
                
                # Update statistics
                self.update_statistics(confidence, has_instruction)
                
                # Log classification
                if self.config.enable_logging and (time.time() - last_log_time) >= log_interval:
                    manual_review = confidence < 0.5
                    self.logger.log_classification(
                        predicted_class=predicted_class,
                        confidence=confidence,
                        bin_location=bin_location,
                        verification_status=verification_status,
                        manual_review=manual_review
                    )
                    last_log_time = time.time()
                
                # Render dashboard
                frame = self.dashboard.render_inference_info(
                    frame, predicted_class, confidence, verification_status, status_color
                )
                
                if has_instruction and self.config.enable_sorting:
                    frame = self.dashboard.render_sorting_info(
                        frame, sorting_instruction, bin_location, bin_color
                    )
                
                # Display FPS
                if self.config.display_fps:
                    cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Render statistics if enabled
            if self.show_stats:
                frame = self.dashboard.render_statistics(frame, self.stats)
            
            # Render keyboard controls
            frame = self.dashboard.render_keyboard_controls(frame)
            
            cv2.imshow('Waste Classification System', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.show_stats = not self.show_stats
                logger.info(f"Statistics display: {'ON' if self.show_stats else 'OFF'}")
            elif key == ord('r'):
                self.stats = {
                    'total_items_sorted': 0,
                    'high_confidence_sorts': 0,
                    'medium_confidence_sorts': 0,
                    'manual_reviews_required': 0,
                    'total_instructions': 0
                }
                logger.info("Statistics reset")
            elif key == ord('e'):
                # Empty all bins
                self.sorting_simulator.empty_bins()
                logger.info("All bins emptied")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save sorting log before shutdown
        if self.config.enable_logging:
            self.sorting_simulator.save_log()
            logger.info("Sorting log saved")
        
        logger.info("Application shutdown complete")
    
    def shutdown(self):
        """Cleanup resources"""
        logger.info("Shutting down application...")

if __name__ == "__main__":
    # Create default configuration if it doesn't exist
    config_path = "config.json"
    if not Path(config_path).exists():
        config = AppConfig()
        config.save_to_file(config_path)
        logger.info(f"Created default configuration file: {config_path}")
    else:
        config = AppConfig.load_from_file(config_path)
        logger.info(f"Loaded configuration from: {config_path}")
    
    # Create log directory
    Path("logs").mkdir(exist_ok=True)
    
    # Create and run application
    app = RealTimeClassificationApp(config)
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    finally:
        app.shutdown()

