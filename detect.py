import cv2
import numpy as np
import random
import os
from typing import List, Dict, Tuple

class WepScanDetector:
    """
    WepScan weapon detection simulator using mock YOLO-style detection results.
    This class simulates the behavior of a trained YOLOv8 model for demonstration purposes.
    """
    
    def __init__(self):
        """Initialize the detector with weapon categories and confidence parameters"""
        self.weapon_categories = [
            'gun', 'pistol', 'rifle', 'knife', 'blade', 'explosive', 
            'grenade', 'suspicious_object', 'metal_object', 'sharp_object'
        ]
        
        # Confidence thresholds
        self.alert_threshold = 0.5
        self.min_confidence = 0.3
        self.max_confidence = 0.9
        
        # Color mapping for different threat levels
        self.colors = {
            'high': (0, 0, 255),    # Red for high threat
            'medium': (0, 165, 255), # Orange for medium threat
            'low': (0, 255, 255),    # Yellow for low threat
            'safe': (0, 255, 0)      # Green for safe objects
        }
    
    def detect_weapons(self, image_path: str) -> Dict:
        """
        Simulate weapon detection on an X-ray image.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            Dict: Detection results including bounding boxes, labels, and confidence scores
        """
        try:
            # Load and validate image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            height, width = image.shape[:2]
            
            # Generate mock detections
            detections = self._generate_mock_detections(width, height)
            
            # Determine threat level and alert status
            threat_level = self._calculate_threat_level(detections)
            alert_triggered = any(d['confidence'] >= self.alert_threshold for d in detections)
            
            return {
                'detections': detections,
                'threat_level': threat_level,
                'alert_triggered': alert_triggered,
                'image_dimensions': (width, height)
            }
            
        except Exception as e:
            raise Exception(f"Detection failed: {str(e)}")
    
    def _generate_mock_detections(self, width: int, height: int) -> List[Dict]:
        """
        Generate realistic mock detection results.
        
        Args:
            width (int): Image width
            height (int): Image height
            
        Returns:
            List[Dict]: List of detection dictionaries
        """
        detections = []
        
        # Randomly determine number of detections (0-4 objects)
        num_detections = random.randint(0, 4)
        
        for _ in range(num_detections):
            # Random weapon category
            label = random.choice(self.weapon_categories)
            
            # Generate realistic confidence based on weapon type
            if label in ['gun', 'pistol', 'rifle', 'knife', 'explosive']:
                # Higher confidence for clear weapons
                confidence = random.uniform(0.6, 0.9)
            else:
                # Lower confidence for suspicious objects
                confidence = random.uniform(0.3, 0.7)
            
            # Generate realistic bounding box
            box_width = random.randint(50, min(200, width // 3))
            box_height = random.randint(30, min(150, height // 3))
            
            x1 = random.randint(0, max(1, width - box_width))
            y1 = random.randint(0, max(1, height - box_height))
            x2 = x1 + box_width
            y2 = y1 + box_height
            
            detection = {
                'label': label,
                'confidence': round(confidence, 3),
                'bbox': [x1, y1, x2, y2],
                'color': self._get_color_for_confidence(confidence)
            }
            
            detections.append(detection)
        
        return detections
    
    def _calculate_threat_level(self, detections: List[Dict]) -> str:
        """
        Calculate overall threat level based on detections.
        
        Args:
            detections (List[Dict]): List of detections
            
        Returns:
            str: Threat level (CRITICAL, HIGH, MEDIUM, LOW, SAFE)
        """
        if not detections:
            return "SAFE"
        
        max_confidence = max(d['confidence'] for d in detections)
        high_threat_items = ['gun', 'pistol', 'rifle', 'explosive', 'grenade']
        
        # Check for high-threat items with high confidence
        critical_detections = [
            d for d in detections 
            if d['label'] in high_threat_items and d['confidence'] >= 0.8
        ]
        
        if critical_detections:
            return "CRITICAL"
        elif max_confidence >= 0.7:
            return "HIGH"
        elif max_confidence >= 0.5:
            return "MEDIUM"
        elif max_confidence >= 0.3:
            return "LOW"
        else:
            return "SAFE"
    
    def _get_color_for_confidence(self, confidence: float) -> Tuple[int, int, int]:
        """
        Get color based on confidence level.
        
        Args:
            confidence (float): Detection confidence
            
        Returns:
            Tuple[int, int, int]: BGR color tuple
        """
        if confidence >= 0.7:
            return self.colors['high']
        elif confidence >= 0.5:
            return self.colors['medium']
        elif confidence >= 0.3:
            return self.colors['low']
        else:
            return self.colors['safe']
    
    def draw_detections(self, image_path: str, detections: List[Dict], output_path: str):
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            image_path (str): Path to input image
            detections (List[Dict]): Detection results
            output_path (str): Path to save annotated image
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Draw each detection
            for detection in detections:
                bbox = detection['bbox']
                label = detection['label']
                confidence = detection['confidence']
                color = detection['color']
                
                # Draw bounding box
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Prepare label text
                label_text = f"{label}: {confidence:.2f}"
                
                # Calculate text size and position
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, font, font_scale, thickness
                )
                
                # Draw label background
                label_x = bbox[0]
                label_y = bbox[1] - 10
                if label_y < text_height:
                    label_y = bbox[1] + text_height + 10
                
                cv2.rectangle(
                    image,
                    (label_x, label_y - text_height - baseline),
                    (label_x + text_width, label_y + baseline),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    image, label_text, (label_x, label_y - baseline),
                    font, font_scale, (255, 255, 255), thickness
                )
            
            # Save annotated image
            success = cv2.imwrite(output_path, image)
            if not success:
                raise ValueError(f"Failed to save image to {output_path}")
                
        except Exception as e:
            raise Exception(f"Failed to draw detections: {str(e)}")

# Example usage and testing functions
def test_detector():
    """Test function for the detector"""
    detector = WepScanDetector()
    
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_path = "test_xray.jpg"
    cv2.imwrite(test_path, test_image)
    
    try:
        # Run detection
        results = detector.detect_weapons(test_path)
        print("Detection Results:")
        print(f"Threat Level: {results['threat_level']}")
        print(f"Alert Triggered: {results['alert_triggered']}")
        print(f"Number of detections: {len(results['detections'])}")
        
        for i, detection in enumerate(results['detections']):
            print(f"  Detection {i+1}: {detection['label']} ({detection['confidence']:.3f})")
        
        # Draw detections
        output_path = "test_output.jpg"
        detector.draw_detections(test_path, results['detections'], output_path)
        print(f"Annotated image saved to: {output_path}")
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        # Clean up test files
        if os.path.exists(test_path):
            os.remove(test_path)

if __name__ == "__main__":
    test_detector()
