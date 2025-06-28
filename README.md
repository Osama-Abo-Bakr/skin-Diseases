# Skin Diseases Detection App

A Streamlit web application for detecting skin diseases using YOLO (You Only Look Once) object detection model.    

## Features

- Upload skin images for disease detection
- Displays prediction with confidence percentage
- Mobile-friendly interface
- Disclaimer for medical use

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.7 or higher
- pip package manager

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/skin-disease-detection.git
   cd skin-disease-detection
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install system dependencies (for Linux):
   ```bash
   sudo apt-get install -y libgl1
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run main.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload a skin image using the sidebar uploader

4. View the prediction results

## File Structure

```
skin-disease-detection/
├── main.py                # Main application code
├── requirements.txt       # Python dependencies
├── packages.txt          # System dependencies
└── models/               # Contains the YOLO model (not included in repo)
```

## Model Information

The application uses a pre-trained YOLO model (`yolo-medical_30epoch.pt`) specifically fine-tuned for medical skin disease detection with 30 epochs of training.

## Important Note

⚠️ **Disclaimer**: All AI-generated diagnoses and advice are for informational purposes only and should not replace professional medical consultation. Always consult with a qualified healthcare provider for medical advice.

## Contact

- **Developer**: Osama Abo Bakr
- **Phone**: +20-1274011748