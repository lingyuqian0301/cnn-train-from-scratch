#dataset we use to train
1. [Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) by Mostafa Abla

# How to Run the Waste Classification System

## Prerequisites

1. **Python 3.7+** installed
2. **Webcam** connected to your computer
3. **Trained model** at `saved_models/best_model.pth` (already exists ✓)

## Step 1: Install Dependencies

Make sure you have all required packages installed. If you're using a virtual environment, activate it first:

```bash
# Activate virtual environment (if using one)
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install/update requirements
pip install -r requirement.txt
```

## Step 2: Run the Application

Simply run the main integration app:

```bash
python main_app.py
```

## What Happens When You Run

1. **Configuration**: The app will load `config.json` (or create a default one if it doesn't exist)
2. **Model Loading**: It loads your trained model from `saved_models/best_model.pth`
3. **Webcam**: Opens your webcam feed
4. **Dashboard**: Displays real-time classification with:
   - Classification results
   - Confidence scores
   - Sorting instructions
   - Statistics panel
   - Keyboard controls help

## Keyboard Controls

While the application is running:

- **`q`**: Quit the application
- **`s`**: Toggle statistics display on/off
- **`r`**: Reset statistics counter

## Output Files

The application creates:

- **`logs/classifications.csv`**: Logs all classifications with timestamps, confidence, bin assignments, etc.
- **`config.json`**: Configuration file (auto-created if missing)

## Troubleshooting

### Webcam Not Opening
- Make sure your webcam is connected and not being used by another application
- Check camera permissions in Windows Settings

### Model Not Found Error
- Verify that `saved_models/best_model.pth` exists
- Check the model path in `config.json`

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirement.txt`
- Make sure `object-detection.py` is in the same directory as `main_app.py`

### Low FPS
- The app targets 20 FPS by default
- You can adjust `fps_target` in `config.json`
- Lower resolution or disable some features if needed

## Alternative: Run Original Object Detection

If you want to run the simpler version without the dashboard:

```bash
python object-detection.py
```

This uses the original `WebcamPredictor` class directly.

