# Lip-reader-based-on-LRW Dataset
Deep learning-based lip reading system using ResNet + Transformer to predict spoken words from silent video (LRW dataset).

This project presents a deep learning-based Visual Speech Recognition (Lip Reading) system that predicts spoken words using only lip movements from video, without relying on audio. Built on the LRW (Lip Reading in the Wild) dataset, the system combines a ResNet-18 model for spatial feature extraction with a Transformer encoder for temporal sequence modeling across video frames.

The pipeline includes preprocessing of video data into efficient .npy format, robust training, and a real-time webcam inference module that displays predicted words with confidence scores.

🎥 Demo Video

🔗 Demo:https://drive.google.com/file/d/1cTRRDjQ0H77NhYmk_Vj19uZ27DFdKR7K/view?usp=sharing

🔗 Download Best Model (.pt):
[](https://www.kaggle.com/datasets/aryanchoubey4842/trained-model-for-lrw-dataset)

Includes trained weights (best_model.pt) for direct inference without training.

🚀 Features
End-to-end pipeline
ResNet-18 + Transformer model
Efficient preprocessing (.npy format)
Real-time webcam prediction
Top-K predictions with confidence

🧱 Project Structure
Lipreading_using_Temporal_Convolutional_Networks/

├── dataset.py
├── preprocess.py
├── model.py
├── train.py
├── webcam.py
├── best_model.pt

📊 Results

Training Accuracy: 81.9%
Validation Accuracy: 75.3%

📈 Training Curves

🔹 Accuracy Curve
<img width="900" height="325" alt="image" src="https://github.com/user-attachments/assets/a8193786-0e6f-4e3c-9519-758e3b2552a3" />

🔹 Loss Curve
<img width="900" height="325" alt="image" src="https://github.com/user-attachments/assets/872a62e5-e060-4720-992f-4cb22f3dfd59" />


⚙️ Preprocessing

python preprocess.py
Extract frames
Grayscale conversion
Mouth cropping (88×88)
Save as .npy

🧠 Model Architecture

ResNet-18 → spatial features
Transformer Encoder → temporal learning
Global Average Pooling
Fully Connected Layer (500 classes)

🏋️ Training

python train.py
AdamW optimizer
Cosine Annealing scheduler
Gradient clipping

🎥 Real-Time Inference

python webcam.py
Webcam-based prediction
29-frame buffering
Top predictions with confidence

🧪 Tech Stack

Python
PyTorch
OpenCV
NumPy

📂 Dataset

LRW (Lip Reading in the Wild)
500 word classes
BBC broadcast videos
© Dataset Credits

The LRW dataset is derived from BBC broadcast recordings.

© British Broadcasting Corporation (BBC)
Introduced by Chung & Zisserman (2016)
Used for research and educational purposes only

⚠️ Limitations

Word-level prediction only
No language context
Fixed cropping

🔮 Future Work

Sentence-level lip reading
Better backbone (ResNet-50)
Landmark-based detection
Audio-visual fusion

👥 Contributors

Aryan Choubey
Kriti Gupta 
Lavanya Karna 
Meshwa Verma 
Shrikant Deepak Rudrawar

📄 Full report:https://drive.google.com/file/d/1cmgKsHroQiSrLmK9h9OgydfAif-a5lMU/view?usp=sharing

⭐ Support

If you like this project, give it a ⭐ on GitHub!
