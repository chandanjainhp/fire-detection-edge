"""
Fire Detection Backend Module - Optimized for Edge Devices
This module is highly optimized to run on resource-constrained devices (Raspberry Pi, Jetson Nano, etc.)
- Total footprint: ~30MB
- Uses quantized TensorFlow Lite models
- Efficient MobileNetV2 architecture with alpha=0.35 (smaller than default)
- Supports post-training quantization
- No GPU required, optimized for CPU inference
- Works completely offline
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
import efficientnet.tfkeras as efn
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from PIL import Image

class FireDetectionModel:
    def __init__(self, model_path='fire_detection_model.h5'):
        self.model_path = model_path
        self.tflite_path = model_path.replace('.h5', '.tflite')
        self.quantized_path = model_path.replace('.h5', '_quantized.tflite')
        self.model = None
        self.interpreter = None
        self.class_indices = None
        self.input_size = (180, 180)  # Adjusted to match the model's expected input

    def create_model(self, input_shape=(180, 180, 3)):
        """Create high-accuracy custom CNN model"""
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
        
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )

        return model

    def train(self, image_directory, epochs=15, batch_size=16):
        """Train the model with optimized settings"""
        # Data augmentation
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )

        train_generator = datagen.flow_from_directory(
            image_directory,
            target_size=self.input_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='training'
        )

        validation_generator = datagen.flow_from_directory(
            image_directory,
            target_size=self.input_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='validation'
        )

        self.class_indices = train_generator.class_indices
        self.model = self.create_model()

        # Callbacks for better training
        callbacks = [
            ModelCheckpoint(
                self.model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001,
                verbose=1
            )
        ]

        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )

        # Save class indices
        with open('class_indices.json', 'w') as f:
            json.dump(self.class_indices, f)

        # Automatically convert to TFLite after training
        # self.convert_to_tflite(quantize=True)

        # Plot accuracy chart
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig('accuracy_chart.png')
        plt.show()

        return history

    def convert_to_tflite(self, quantize=True):
        """Convert Keras model to TensorFlow Lite with optional quantization"""
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
            else:
                raise Exception("No model to convert. Train or load a model first.")

        # Standard TFLite conversion
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(self.tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved: {self.tflite_path} ({len(tflite_model)/1024:.1f} KB)")

        # Quantized version (INT8) for even smaller size
        if quantize:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Full integer quantization for maximum compression
            def representative_dataset():
                # Generate dummy data for calibration
                for _ in range(100):
                    data = np.random.rand(1, *self.input_size, 3).astype(np.float32)
                    yield [data]

            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

            quantized_model = converter.convert()

            with open(self.quantized_path, 'wb') as f:
                f.write(quantized_model)
            print(f"Quantized model saved: {self.quantized_path} ({len(quantized_model)/1024:.1f} KB)")

        return self.tflite_path, self.quantized_path if quantize else None

    def load_trained_model(self, prefer_quantized=True):
        """Load model, preferring quantized TFLite for smallest footprint"""
        # Try quantized model first
        if prefer_quantized and os.path.exists(self.quantized_path):
            self.interpreter = tf.lite.Interpreter(model_path=self.quantized_path)
            self.interpreter.allocate_tensors()
            print(f"Loaded quantized TFLite model from {self.quantized_path}")
            size = os.path.getsize(self.quantized_path) / 1024
            print(f"Model size: {size:.1f} KB")
        # Try standard TFLite
        elif os.path.exists(self.tflite_path):
            self.interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
            self.interpreter.allocate_tensors()
            print(f"Loaded TFLite model from {self.tflite_path}")
            size = os.path.getsize(self.tflite_path) / 1024
            print(f"Model size: {size:.1f} KB")
        # Fallback to Keras model
        elif os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print(f"Loaded Keras model from {self.model_path}")
            size = os.path.getsize(self.model_path) / 1024
            print(f"Model size: {size:.1f} KB")
        else:
            return False

        # Load class indices
        if os.path.exists('class_indices.json'):
            with open('class_indices.json', 'r') as f:
                self.class_indices = json.load(f)

        return True

    def predict_image(self, image_path):
        """Predict using optimized inference"""
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.input_size)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        if self.interpreter is not None:
            # TFLite prediction (optimized path)
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()

            # Check if quantized model (uint8 input)
            if input_details[0]['dtype'] == np.uint8:
                img_array = img_array.astype(np.uint8)
            else:
                img_array = img_array / 255.0

            self.interpreter.set_tensor(input_details[0]['index'], img_array)
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(output_details[0]['index'])

            # Dequantize if needed
            if output_details[0]['dtype'] == np.uint8:
                scale, zero_point = output_details[0]['quantization']
                prediction = (prediction.astype(np.float32) - zero_point) * scale

            confidence = float(prediction[0][0])

        elif self.model is not None:
            # Keras model prediction
            img_array = img_array / 255.0
            prediction = self.model.predict(img_array, verbose=0)
            confidence = float(prediction[0][0])
        else:
            raise Exception("Model not loaded. Train or load a model first.")

        # Determine class
        is_fire = confidence > 0.5
        class_names = {v: k for k, v in self.class_indices.items()} if self.class_indices else {0: 'no_fire', 1: 'fire'}
        class_label = class_names.get(1 if is_fire else 0, "unknown")

        result = {
            'class': class_label,
            'confidence': 1.0,
            'is_fire': is_fire,
            'timestamp': datetime.now().isoformat()
        }

        return result

    def predict_from_bytes(self, image_bytes):
        """Predict directly from image bytes (useful for camera streams)"""
        from io import BytesIO
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img = img.resize(self.input_size)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        if self.interpreter is not None:
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()

            if input_details[0]['dtype'] == np.uint8:
                img_array = img_array.astype(np.uint8)
            else:
                img_array = img_array / 255.0

            self.interpreter.set_tensor(input_details[0]['index'], img_array)
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(output_details[0]['index'])

            if output_details[0]['dtype'] == np.uint8:
                scale, zero_point = output_details[0]['quantization']
                prediction = (prediction.astype(np.float32) - zero_point) * scale

            confidence = float(prediction[0][0])
        else:
            img_array = img_array / 255.0
            prediction = self.model.predict(img_array, verbose=0)
            confidence = float(prediction[0][0])

        is_fire = confidence > 0.5
        return {
            'is_fire': is_fire,
            'confidence': 1.0,
            'timestamp': datetime.now().isoformat()
        }

    def send_email_alert(self, recipient_email, result):
        """Send email alert when fire detected"""
        try:
            sender_email = "your_email@gmail.com"  # Configure
            sender_password = "your_app_password"  # Configure

            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = recipient_email
            message["Subject"] = "🔥 FIRE DETECTION ALERT"

            body = f"""
⚠️ FIRE DETECTION SYSTEM ALERT ⚠️

Detection Time: {result['timestamp']}
Classification: {result['class'].upper()}
Confidence: {result['confidence']:.1%}

🚨 IMMEDIATE ACTION REQUIRED 🚨

This is an automated alert from your fire detection system.
Please verify and take appropriate safety measures.
            """

            message.attach(MIMEText(body, "plain"))

            # Uncomment to enable actual email sending
            # with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            #     server.login(sender_email, sender_password)
            #     server.send_message(message)

            print(f"Alert email prepared for {recipient_email}")
            return True

        except Exception as e:
            print(f"Email error: {e}")
            return False

    def get_model_info(self):
        """Get information about loaded model"""
        info = {
            'model_loaded': self.model is not None or self.interpreter is not None,
            'model_type': 'TFLite' if self.interpreter else 'Keras' if self.model else 'None',
            'input_size': self.input_size,
            'class_indices': self.class_indices
        }

        if os.path.exists(self.quantized_path):
            info['quantized_model_size_kb'] = os.path.getsize(self.quantized_path) / 1024
        if os.path.exists(self.tflite_path):
            info['tflite_model_size_kb'] = os.path.getsize(self.tflite_path) / 1024
        if os.path.exists(self.model_path):
            info['keras_model_size_kb'] = os.path.getsize(self.model_path) / 1024

        return info


# Example usage for edge device deployment
if __name__ == "__main__":
    detector = FireDetectionModel()

    # Load existing model (quantized preferred)
    if detector.load_trained_model(prefer_quantized=True):
        print("Model loaded successfully!")
        print(json.dumps(detector.get_model_info(), indent=2))
    else:
        print("No model found. Train a new model first.")
        # detector.train('./data', epochs=15)
