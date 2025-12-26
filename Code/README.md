# 🔥 Fire Detection System - Modern Web Interface

A modern, AI-powered fire detection system with a sleek web interface built using TensorFlow, Flask, and Tailwind CSS.

## Features

✨ **Modern UI/UX**
- Responsive design with Tailwind CSS
- Drag-and-drop image upload
- Real-time predictions
- Beautiful animations and transitions

🤖 **AI-Powered Detection**
- EfficientNetV2 model with transfer learning
- High accuracy fire detection
- Confidence scores
- Automatic model training on startup

📊 **Analytics Dashboard**
- Live statistics
- Detection history
- Visual results display

🔔 **Alert System**
- Email notifications for fire detection
- Customizable alert settings

🎯 **Training Module**
- Train new models through the UI
- Adjustable hyperparameters
- Progress tracking

## Project Structure

```
fire-detection-system/
├── app.py                          # Flask API server
├── fire_detection_backend.py       # ML model backend
├── templates/
│   └── index.html                  # Modern web interface
├── uploads/                        # Temporary upload folder
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Organize your training data in the following structure:

```
/working/data/
├── fire/
│   ├── fire_image1.jpg
│   ├── fire_image2.jpg
│   └── ...
└── no_fire/
    ├── normal_image1.jpg
    ├── normal_image2.jpg
    └── ...
```

### 3. Prepare Training Data (Optional)

If you want to train a custom model, place images in `./data/` with subfolders:
- `data/fire/` - Images with fire
- `data/no_fire/` - Images without fire

The model will train automatically on startup if data exists and no model is found.

### 4. Run the Application

**Option A: Using Docker (Recommended)**

```bash
docker-compose up --build
```

**Option B: Local Development**

```bash
python app.py
```

The server will start on `http://localhost:5000`

### 5. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

### 6. Monitoring

Basic monitoring metrics are available at: `http://localhost:5000/metrics`

## Usage

### Making Predictions

1. **Upload Image**: 
   - Drag and drop an image onto the upload zone, or
   - Click "Browse Files" to select an image

2. **Enable Alerts** (Optional):
   - Check "Enable Email Alerts"
   - Enter recipient email address
   - Configure SMTP settings in `fire_detection_backend.py`

3. **Analyze**:
   - Click "Analyze Image" button
   - Wait for results (usually 1-2 seconds)

4. **View Results**:
   - Classification (Fire/No Fire)
   - Confidence score
   - Visual indicators
   - Detection history

### Running Multiple Predictions

The model stays loaded in memory, so you can:
- Upload and analyze as many images as you want
- No need to retrain between predictions
- Fast inference (1-2 seconds per image)
- Statistics are tracked automatically

## API Endpoints

### GET /api/status
Check system status and model availability

### POST /api/predict
Predict fire detection for uploaded image

**Form Data:**
- `image`: Image file (required)
- `send_alert`: Enable email alert (optional)
- `alert_email`: Email address for alerts (optional)

**Response:**
```json
{
  "class": "fire",
  "confidence": 0.95,
  "is_fire": true,
  "timestamp": "2025-12-26T15:23:00",
  "alert_sent": true
}
```

### POST /api/train
Train or retrain the model

**JSON Body:**
```json
{
  "data_directory": "/working/data",
  "epochs": 10
}
```

## Email Alert Configuration

To enable email alerts, edit `fire_detection_backend.py`:

```python
def send_email_alert(self, recipient_email, result):
    # Configure SMTP settings
    sender_email = "your_email@gmail.com"
    sender_password = "your_app_password"  # Use app password for Gmail

    # Uncomment the SMTP code block
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.send_message(message)
```

### Gmail Setup:
1. Enable 2-factor authentication
2. Generate an app password
3. Use the app password in the code

## Model Architecture

```
Conv2D(32) -> MaxPooling2D
Conv2D(64) -> MaxPooling2D
Conv2D(128) -> MaxPooling2D
Flatten
Dense(512) -> Dropout(0.5)
Dense(1, sigmoid)
```

## Customization

### Change Image Size
Edit `fire_detection_backend.py`:
```python
target_size=(180, 180)  # Change to your preferred size
```

### Adjust Model Architecture
Modify the `create_model()` method in `fire_detection_backend.py`

### Customize UI Colors
Edit the Tailwind classes in `templates/index.html`

## Troubleshooting

**Model Not Loading:**
- Ensure `fire_detection_model.h5` exists
- Check `class_indices.json` is present
- Retrain the model if files are missing

**Prediction Errors:**
- Verify image format (JPG, PNG, JPEG, GIF)
- Check file size (max 16MB)
- Ensure model is trained

**Email Not Sending:**
- Configure SMTP settings correctly
- Use app password for Gmail
- Check firewall settings

## Performance Tips

- Use GPU for faster training (TensorFlow will auto-detect)
- Reduce image size for faster inference
- Increase batch size if you have more RAM
- Use data augmentation for better accuracy

## License

MIT License - feel free to use and modify!

## Support

For issues or questions, please create an issue on the repository.

---

Built with ❤️ using TensorFlow, Flask, and Tailwind CSS
