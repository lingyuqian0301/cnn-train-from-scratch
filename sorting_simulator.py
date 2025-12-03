"""
Sorting Simulator - Person 2
Generates sorting instructions, tracks bin capacity, calculates statistics
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinLocation(Enum):
    """Bin location enumeration"""
    BIN_A = "Bin A (Recyclables)"
    BIN_B = "Bin B (Hazardous)"
    BIN_C = "Bin C (Organic)"
    BIN_D = "Bin D (Textiles)"
    BIN_E = "Bin E (General Waste)"

@dataclass
class SortingInstruction:
    """Sorting instruction data structure"""
    instruction_text: str
    bin_location: str
    confidence: float
    requires_manual_review: bool
    predicted_class: str
    timestamp: str

class SortingSimulator:
    """Simulates waste sorting mechanism with bin capacity tracking"""
    
    def __init__(self, max_bin_capacity: int = 100, log_file: str = "logs/sorting_log.json"):
        """
        Initialize sorting simulator
        
        Args:
            max_bin_capacity: Maximum items per bin before warning
            log_file: Path to JSON log file
        """
        self.max_bin_capacity = max_bin_capacity
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Waste-to-bin mapping (5 types → 5 bins)
        self.waste_to_bin = {
            'glass': BinLocation.BIN_A,
            'green-glass': BinLocation.BIN_A,
            'brown-glass': BinLocation.BIN_A,
            'white-glass': BinLocation.BIN_A,
            'metal': BinLocation.BIN_B,
            'plastic': BinLocation.BIN_C,
            'paper': BinLocation.BIN_D,
            'cardboard': BinLocation.BIN_D,
            'biological': BinLocation.BIN_E,
            'battery': BinLocation.BIN_B,  # Hazardous
            'trash': BinLocation.BIN_E,
            'clothes': BinLocation.BIN_E,
            'shoes': BinLocation.BIN_E,
        }
        
        # Bin capacity tracking
        self.bin_counts = {
            BinLocation.BIN_A: 0,
            BinLocation.BIN_B: 0,
            BinLocation.BIN_C: 0,
            BinLocation.BIN_D: 0,
            BinLocation.BIN_E: 0,
        }
        
        # Statistics
        self.stats = {
            'total_instructions': 0,
            'auto_sorted': 0,
            'needs_verification': 0,
            'manual_reviews': 0,
            'bin_warnings': 0,
        }
        
        # Operation log
        self.operations_log: List[Dict] = []
        
        # Color mapping for display (BGR format for OpenCV)
        self.bin_colors = {
            BinLocation.BIN_A: (0, 255, 0),      # Green
            BinLocation.BIN_B: (0, 0, 255),       # Red
            BinLocation.BIN_C: (0, 165, 255),    # Orange
            BinLocation.BIN_D: (255, 0, 255),     # Magenta
            BinLocation.BIN_E: (128, 128, 128)   # Gray
        }
        
        logger.info("Sorting Simulator initialized")
    
    def get_bin_location(self, waste_type: str) -> Optional[BinLocation]:
        """Get bin location for a waste type"""
        return self.waste_to_bin.get(waste_type.lower())
    
    def get_bin_color(self, waste_type: str) -> tuple:
        """
        Get bin color for display (BGR format for OpenCV)
        
        Args:
            waste_type: Waste class name
            
        Returns:
            Tuple of (B, G, R) color values
        """
        bin_location = self.get_bin_location(waste_type)
        if bin_location:
            return self.bin_colors.get(bin_location, (128, 128, 128))
        return (128, 128, 128)  # Default gray
    
    def get_verification_status(self, confidence: float) -> tuple:
        """
        Get verification status message and color based on confidence
        
        Args:
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            Tuple of (status_message, color_bgr)
        """
        if confidence >= 0.75:
            return "AUTO CLASSIFIED", (0, 255, 0)  # Green
        elif confidence >= 0.5:
            return "NEEDS VERIFICATION", (0, 165, 255)  # Orange
        else:
            return "MANUAL IDENTIFICATION REQUIRED", (0, 0, 255)  # Red
    
    def generate_instruction(self, predicted_class: str, confidence: float) -> SortingInstruction:
        """
        Generate sorting instruction based on confidence level
        
        Confidence Logic:
        >75%:    Direct sort ("MOVE TO BIN X")
        50-75%:  Verify needed ("MOVE TO BIN X - VERIFY")
        <50%:    Manual review ("MANUAL REVIEW REQUIRED")
        
        Args:
            predicted_class: Predicted waste class
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            SortingInstruction object
        """
        bin_location = self.get_bin_location(predicted_class)
        
        if bin_location is None:
            # Unknown waste type - default to general waste
            bin_location = BinLocation.BIN_E
            logger.warning(f"Unknown waste type: {predicted_class}, defaulting to {bin_location.value}")
        
        # Determine instruction based on confidence
        if confidence >= 0.75:
            # High confidence - direct sort
            instruction_text = f"MOVE TO {bin_location.value.upper()} (Confidence: {confidence:.0%})"
            requires_manual_review = False
            self.stats['auto_sorted'] += 1
        elif confidence >= 0.5:
            # Medium confidence - needs verification
            instruction_text = f"MOVE TO {bin_location.value.upper()} - VERIFY (Confidence: {confidence:.0%})"
            requires_manual_review = False
            self.stats['needs_verification'] += 1
        else:
            # Low confidence - manual review required
            instruction_text = f"MANUAL REVIEW REQUIRED (Confidence: {confidence:.0%})"
            requires_manual_review = True
            self.stats['manual_reviews'] += 1
            bin_location = None  # No bin assignment for manual review
        
        # Update bin capacity if instruction was generated
        if bin_location and not requires_manual_review:
            self.bin_counts[bin_location] += 1
            
            # Check for capacity warning
            if self.bin_counts[bin_location] >= self.max_bin_capacity:
                self.stats['bin_warnings'] += 1
                instruction_text += f" [BIN NEARLY FULL: {self.bin_counts[bin_location]}/{self.max_bin_capacity}]"
                logger.warning(f"{bin_location.value} is nearly full: {self.bin_counts[bin_location]}/{self.max_bin_capacity}")
        
        self.stats['total_instructions'] += 1
        
        instruction = SortingInstruction(
            instruction_text=instruction_text,
            bin_location=bin_location.value if bin_location else "N/A",
            confidence=confidence,
            requires_manual_review=requires_manual_review,
            predicted_class=predicted_class,
            timestamp=datetime.now().isoformat()
        )
        
        # Log operation
        self.operations_log.append(asdict(instruction))
        
        return instruction
    
    def get_statistics(self) -> Dict:
        """Get current statistics"""
        return {
            **self.stats,
            'bin_counts': {
                bin.value: count for bin, count in self.bin_counts.items()
            },
            'bin_capacities': {
                bin.value: {
                    'current': count,
                    'max': self.max_bin_capacity,
                    'percentage': (count / self.max_bin_capacity) * 100
                }
                for bin, count in self.bin_counts.items()
            }
        }
    
    def empty_bins(self):
        """Empty all bins (reset counts)"""
        logger.info("Emptying all bins...")
        for bin_location in self.bin_counts:
            self.bin_counts[bin_location] = 0
        logger.info("All bins emptied")
    
    def save_log(self):
        """Save operations log to JSON file"""
        try:
            log_data = {
                'metadata': {
                    'total_operations': len(self.operations_log),
                    'generated_at': datetime.now().isoformat(),
                    'max_bin_capacity': self.max_bin_capacity
                },
                'statistics': self.get_statistics(),
                'operations': self.operations_log
            }
            
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            logger.info(f"Sorting log saved to {self.log_file}")
        except Exception as e:
            logger.error(f"Failed to save sorting log: {e}")
    
    def load_log(self) -> Optional[Dict]:
        """Load previous operations log from JSON file"""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load sorting log: {e}")
        return None


