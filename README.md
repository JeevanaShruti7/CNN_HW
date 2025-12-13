**CNN Image Classification - Web Deployment** 

This project implements a Convolutional Neural Network (CNN) for image classification using PyTorch, exports the model to ONNX, and deploys it on the web using HTML + JavaScript + ONNX Runtime.
The project can be run fully in-browser via GitHub Pages.


**Project Features**

•	Custom image dataset (dataset_CNN_HW)

•	Train/validation/test split

•	CNN model built and trained using PyTorch

•	Training curves and evaluation metrics

•	Confusion matrix visualization

•	Exported to ONNX (cnn_model.onnx)

•	Browser-based model inference using
    o	ONNX Runtime Web
    o	JavaScript

•	Fully hosted using GitHub Pages


**Project Structure**

my_cnn_web/
  -> model/
    cnn_model.onnx
  -> web/
    index.html
    script.js
  -> notebook/
    cnn_hw_code.py
  -> images/
    confusion_matrix.png
    predicted_image.png
  -> README.md
  


**Model Training**

The model is trained using PyTorch with:

•	2 convolution layers

•	ReLU activations

•	MaxPooling

•	Fully connected classifier

•	Cross-entropy loss

•	Adam optimizer

Training was done in Google Colab with GPU support.

**Evaluation Results**

•	Training Accuracy: ~91%

•	Validation Accuracy: ~88%

•	Test Accuracy: ~86%

**Additional outputs included:**

•	Confusion Matrix

•	Predicted results


**Web Deployment**

The index.html page loads the ONNX model and runs inference directly in the browser.


**Run Online (GitHub Pages)**

https://github.com/JeevanaShruti7/CNN_HW.git


**How to Run Locally**

1.	Clone this repo:

git clone https://github.com/JeevanaShruti7/CNN_HW.git

2.	Open web/index.html directly in your browser
OR run a local server:
python -m http.server 8000

Then visit:
http://localhost:8000/web/


**Dependencies**

Training (Python)

•	PyTorch

•	NumPy

•	Matplotlib

•	Scikit-Learn

•	ONNX

•	ONNX Runtime


Deployment (Web)

•	ONNX Runtime Web

•	JavaScript

•	HTML/CSS


 **References**
 
•	PyTorch Documentation

•	ONNX Runtime Web Docs

•	Google Colab

•	MDN Web Docs


**Contact**

Jeevana Shruti

