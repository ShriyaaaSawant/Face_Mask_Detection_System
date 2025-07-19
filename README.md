🧠 Face Mask Detection System
A real-time AI-based system that detects whether a person is wearing a face mask or not using webcam input.
Built with Python, OpenCV, TensorFlow, and Keras.

📸 Demo
Live webcam output:
🟩 Displays "Mask" if a face is detected with a mask
🟥 Displays "No Mask" otherwise

🚀 Features

Real-time face detection using OpenCV
Deep learning model trained with MobileNetV2
Custom dataset of masked and unmasked faces
Accurate binary classification (with_mask / without_mask)
Easily extendable for mobile or web deployment

🛠️ Tech Stack
Tool	             Usage
Python	            Core programming language
OpenCV	            Real-time face detection
TensorFlow	        Deep learning backend
Keras          	    High-level model API
MobileNetV2	        Transfer learning model base

🗂️ Dataset
The dataset used contains images of:
People wearing masks
People not wearing masks

Folder structure:

Dataset/
├── with_mask/
├── without_mask/

🧪 How to Run
Step 1: Clone or Download the Project
git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection

Step 2: Install Dependencies
pip install opencv-python tensorflow numpy

Step 3: Train the Model (optional if already trained)
python train_mask_model.py

Step 4: Run the Detection System
python detect_mask_video.py

Press q to quit the webcam window.

 Output Example
Face with Mask → ✅ Detected: Mask
Face without Mask → ❌ Detected: No Mask
📄 Credits
Inspired by open-source Face Mask Detection projects
Dataset compiled from various public image datasets

✨ Author
Shriya Sawant
Final Year Computer Engineering | SPPU
