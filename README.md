# Traffic Sign Recognition

## About the Project

This project is a Traffic Sign Recognition system developed using Python, Computer Vision, and Deep Learning.

A Convolutional Neural Network (CNN) is trained using the GTSRB (German Traffic Sign Recognition Benchmark) dataset to classify traffic signs from images.

## Features

* Traffic sign image classification
* CNN-based deep learning model
* Image preprocessing using OpenCV
* Prediction with confidence score
* Streamlit web application

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Streamlit

## Model Accuracy

The trained model achieved approximately **99.46% accuracy**.

## Project Files

```text
app.py
train_model.py
traffic_sign_model.keras
requirements.txt
README.md
```

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

## Dataset

The project uses the GTSRB (German Traffic Sign Recognition Benchmark) dataset.

The complete dataset is not included in this repository because of its large size.

## Future Improvements

* Real-time traffic sign recognition using a webcam
* Improve recognition under different lighting and weather conditions
* Deploy the application online

## Project Type

Computer Vision / Deep Learning / Data Science
