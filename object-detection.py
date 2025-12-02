import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import time

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SortingMechanism:
    """Simulates a sorting mechanism that maps detected classes to bins"""
    def __init__(self):
        # Define bin mappings based on waste categories
        self.bin_mapping = {
            # Bin A: Recyclables (Paper, Cardboard, Plastic, Glass, Metal)
            'paper': 'Bin A (Recyclables)',
            'cardboard': 'Bin A (Recyclables)',
            'plastic': 'Bin A (Recyclables)',
            'glass': 'Bin A (Recyclables)',
            'metal': 'Bin A (Recyclables)',
            
            # Bin B: Hazardous/Electronics (Battery)
            'battery': 'Bin B (Hazardous)',
            
            # Bin C: Organic/Biological Waste
            'biological': 'Bin C (Organic)',
            
            # Bin D: Textiles (Clothes, Shoes)
            'clothes': 'Bin D (Textiles)',
            'shoes': 'Bin D (Textiles)',
            
            # Bin E: General Trash
            'trash': 'Bin E (General Waste)'
        }
    
    def get_sorting_instruction(self, class_name):
        """Returns the sorting instruction for a given class"""
        bin_assignment = self.bin_mapping.get(class_name.lower(), 'Bin E (General Waste)')
        return f"Move to {bin_assignment}"
    
    def get_bin_color(self, class_name):
        """Returns a color for the bin assignment (BGR format for OpenCV)"""
        bin_assignment = self.bin_mapping.get(class_name.lower(), 'Bin E (General Waste)')
        
        color_map = {
            'Bin A (Recyclables)': (0, 255, 0),      # Green
            'Bin B (Hazardous)': (0, 0, 255),       # Red
            'Bin C (Organic)': (0, 165, 255),        # Orange
            'Bin D (Textiles)': (255, 0, 255),       # Magenta
            'Bin E (General Waste)': (128, 128, 128) # Gray
        }
        
        return color_map.get(bin_assignment, (128, 128, 128))
    
    def get_verification_status(self, confidence):
        """Returns verification status message based on confidence level"""
        if confidence >= 0.75:
            return "AUTO CLASSIFIED", (0, 255, 0)  # Green - Direct classification
        elif confidence >= 0.5:
            return "NEEDS VERIFICATION", (0, 165, 255)  # Orange - Verify before sorting
        else:
            return "MANUAL IDENTIFICATION REQUIRED", (0, 0, 255)  # Red - Manual handling

class WebcamPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load the saved model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint['class_names']

        self.model = CNN(num_classes=len(self.class_names))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize sorting mechanism
        self.sorting_mechanism = SortingMechanism()

    def preprocess_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        return img_tensor

    def predict_frame(self, frame):
        try:
            img_tensor = self.preprocess_image(frame)
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
                predicted_class = self.class_names[predicted_idx]
            return predicted_class, confidence
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, None

    def predict_sample_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return
        predicted_class, confidence = self.predict_frame(image)
        verification_status, status_color = self.sorting_mechanism.get_verification_status(confidence)
        
        print(f"Prediction: {predicted_class}, Confidence: {confidence:.2%}")
        print(f"Status: {verification_status}")
        
        # Show image with prediction
        font = cv2.FONT_HERSHEY_SIMPLEX
        display = image.copy()
        
        if confidence >= 0.5:
            sorting_instruction = self.sorting_mechanism.get_sorting_instruction(predicted_class)
            bin_color = self.sorting_mechanism.get_bin_color(predicted_class)
            print(f"Sorting Instruction: {sorting_instruction}")
            
            cv2.putText(display, f"Class: {predicted_class}", (20, 40), font, 0.5, (0, 255, 0), 2)
            cv2.putText(display, f"Confidence: {confidence:.2%}", (20, 80), font, 0.5, (0, 255, 0), 2)
            cv2.putText(display, verification_status, (20, 120), font, 0.5, status_color, 2)
            cv2.putText(display, sorting_instruction, (20, 160), font, 0.5, bin_color, 2)
        else:
            cv2.putText(display, f"Predicted: {predicted_class}", (20, 40), font, 0.5, (0, 0, 255), 2)
            cv2.putText(display, f"Confidence: {confidence:.2%}", (20, 80), font, 0.5, (0, 0, 255), 2)
            cv2.putText(display, verification_status, (20, 120), font, 0.5, status_color, 2)
        
        cv2.imshow("Sample Image Prediction", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def start_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("Webcam started. Press 'q' to quit.")

        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            fps_frame_count += 1
            if fps_frame_count >= 30:
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0

            predicted_class, confidence = self.predict_frame(frame)

            if predicted_class is not None:
                # Get verification status based on confidence
                verification_status, status_color = self.sorting_mechanism.get_verification_status(confidence)
                
                overlay = frame.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                if confidence >= 0.75:
                    # High confidence (>= 0.75): Direct classification
                    sorting_instruction = self.sorting_mechanism.get_sorting_instruction(predicted_class)
                    bin_color = self.sorting_mechanism.get_bin_color(predicted_class)
                    
                    cv2.rectangle(overlay, (10, 10), (550, 170), (0, 0, 0), -1)
                    alpha = 0.7
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                    
                    # Display classification info
                    cv2.putText(frame, f"Class: {predicted_class}", (20, 40), font, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 70), font, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, verification_status, (20, 100), font, 0.7, status_color, 2)
                    cv2.putText(frame, sorting_instruction, (20, 130), font, 0.8, bin_color, 2)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 160), font, 0.6, (0, 255, 0), 2)
                    
                    # Print sorting instruction to console (for logging/simulation)
                    if fps_frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                        print(f"[SORTING] {predicted_class} -> {sorting_instruction} (Confidence: {confidence:.2%}) - AUTO CLASSIFIED")
                
                elif confidence >= 0.5:
                    # Medium confidence (0.5 to 0.75): Needs verification
                    sorting_instruction = self.sorting_mechanism.get_sorting_instruction(predicted_class)
                    bin_color = self.sorting_mechanism.get_bin_color(predicted_class)
                    
                    cv2.rectangle(overlay, (10, 10), (550, 170), (0, 0, 0), -1)
                    alpha = 0.7
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                    
                    # Display classification info with verification warning
                    cv2.putText(frame, f"Class: {predicted_class}", (20, 40), font, 0.8, (0, 165, 255), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 70), font, 0.8, (0, 165, 255), 2)
                    cv2.putText(frame, verification_status, (20, 100), font, 0.7, status_color, 2)
                    cv2.putText(frame, sorting_instruction, (20, 130), font, 0.8, bin_color, 2)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 160), font, 0.6, (0, 255, 0), 2)
                    
                    # Print sorting instruction to console (for logging/simulation)
                    if fps_frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                        print(f"[SORTING] {predicted_class} -> {sorting_instruction} (Confidence: {confidence:.2%}) - NEEDS VERIFICATION")
                
                else:
                    # Low confidence (< 0.5): Manual identification required
                    cv2.rectangle(overlay, (10, 10), (550, 130), (0, 0, 0), -1)
                    alpha = 0.7
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                    
                    # Display manual identification warning
                    cv2.putText(frame, f"Predicted: {predicted_class}", (20, 40), font, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 70), font, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, verification_status, (20, 100), font, 0.7, status_color, 2)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 130), font, 0.6, (0, 255, 0), 2)
                    
                    # Print manual identification requirement to console
                    if fps_frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                        print(f"[SORTING] {predicted_class} (Confidence: {confidence:.2%}) - MANUAL IDENTIFICATION REQUIRED")

            cv2.imshow('Webcam Prediction', frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    model_path = "saved_models/best_model.pth"
    predictor = WebcamPredictor(model_path)

    # Predict from a sample image, uncomment the line below:
    # sample_image_path = "dataset/train/battery/battery22.jpg"
    # predictor.predict_sample_image(sample_image_path)

    # Or, to use the webcam, uncomment the line below:
    predictor.start_webcam()

if __name__ == "__main__":
    main()
