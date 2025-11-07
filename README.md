🧠 Early Identification of Critical Health Conditions of Breast, Chest, and Brain through Medical Imaging
📄 Project Overview

This project aims to assist healthcare professionals in early detection of critical health conditions — specifically tumors and abnormalities in the brain, breast, and chest — using medical imaging techniques powered by deep learning.
By combining Convolutional Neural Networks (CNNs), VGG16 (transfer learning), and 3D U-Net segmentation, this system automates classification and segmentation of medical images, improving diagnostic accuracy and reducing human dependency.

🎯 Objectives

Automate disease detection: Identify and classify tumors in MRI, CT, and mammography images.

Apply deep learning: Implement CNN, VGG16, and 3D U-Net models for accurate prediction.

Transfer learning: Use pre-trained models to improve efficiency on limited medical datasets.

Develop GUI: Provide a simple and interactive user interface for clinicians to upload images and view diagnostic results.

🏗️ System Architecture

Modules include:

Data Preprocessing:

Image resizing, normalization, and noise reduction.

Feature Extraction:

VGG16 pre-trained model for transfer learning.

Classification:

Custom 16-layer CNN for disease classification.

Segmentation:

3D U-Net for tumor localization and visualization.

User Interface:

Tkinter-based GUI for easy image upload and result display.

⚙️ Technologies Used
Category	Tools / Libraries
Language	Python 3.8+
Deep Learning	TensorFlow, Keras
Image Processing	OpenCV, NumPy, Nibabel
Visualization	Matplotlib, Seaborn
Interface	Tkinter
Data Handling	Pandas, Pickle, Scikit-learn
💻 Installation & Setup
Prerequisites

Python 3.8 or higher

GPU (Recommended: NVIDIA GTX 1650 / RTX 3060 or equivalent)

8GB RAM (16GB preferred)

Step 1: Clone the Repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Prepare Model Files

Ensure the following pre-trained model files exist in the model/ directory:

model/
 ├── cnn_weights.h5
 ├── vgg16_weights.hdf5
 ├── model_per_class.h5
 ├── X.txt.npy
 └── Y.txt.npy

Step 4: Run the Application
python GUI.py

🧩 Features

✅ MRI, CT, and Mammography image support
✅ Multi-organ detection: Brain, Breast, Chest
✅ Tumor segmentation using 3D U-Net
✅ Transfer learning for efficient training
✅ Real-time GUI-based predictions
✅ Performance metrics visualization (accuracy, precision, recall, F1 score)

📊 Model Performance
Model	Accuracy	Precision	Recall	F1 Score
VGG16 (Transfer Learning)	87%	High	Good	0.88
Proposed CNN (16-layer)	93%	Higher	Improved	0.91
3D U-Net	Effective for tumor segmentation with strong Dice Coefficient values			
🧠 Deep Learning Models
1️⃣ VGG16 (Transfer Learning)

Used for feature extraction.

Fine-tuned for medical imaging datasets.

2️⃣ Custom CNN

16-layer model built for classification tasks.

Enhanced accuracy through batch normalization and dropout.

3️⃣ 3D U-Net

Performs volumetric segmentation for MRI/CT data.

Calculates Dice coefficient, precision, and specificity for tumor detection.

🖥️ Graphical User Interface (GUI)

Developed using Tkinter.

Users can:

Upload medical images.

Train or load pre-trained models.

Visualize model performance and confusion matrices.

View segmentation outputs and classified results.

📈 Results

The system achieved 93% accuracy with the proposed CNN model.

VGG16 model reached 87% accuracy.

The 3D U-Net demonstrated precise tumor segmentation with a strong Dice similarity coefficient.

The GUI simplifies image uploads and diagnostic interpretation for clinicians.

🔮 Future Enhancements

Integration with Electronic Health Records (EHR).

Expansion to detect more organ-specific diseases.

Deployment as a web-based clinical application.

Incorporation of federated learning for secure model training across hospitals.
