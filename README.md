# Age-Gender-Detection-with-CNNs 

This project builds a CNN model to predict age and gender from facial images. We use the UTKFace dataset, which includes age, gender, and ethnicity labels. After preprocessing and augmenting the data, MobileNetV2 is used as the base for feature extraction, followed by custom layers for age regression and gender classification. To improve performance, we apply techniques like early stopping, learning rate reduction, and model checkpointing during training.

We also built a simple web app using Streamlit to make the model accessible. OpenCV is integrated into the app to allow real-time predictions using a laptop camera. The app captures a live video feed, detects faces, and runs the model to display the predicted age and gender instantly.

Unfortunately, the trained models are larger than 25MB so they can't be uploaded but you can try running the training Notebook as well.
