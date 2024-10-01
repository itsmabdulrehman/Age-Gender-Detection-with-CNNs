import cv2
import numpy as np
from keras.models import load_model

# Load model
model_age = load_model('./model/model_age.hdf5')
model_gender = load_model('./model/model_gender.hdf5')

# Label
label_gender = ['Male', 'Female']

# Detect Image
def detect_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Load Haar cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #x,y give the top-left corner of the detection box
    #w,h are width and height of the box, given by the cascade classifier
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract and resize the face region (the model expects 50x50 images)
        face_img = cv2.resize(gray[y:y+h, x:x+w], (50, 50))

        # Convert to RGB format
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)

        # Expand dimensions to match model input shape
        face_img = np.expand_dims(face_img, axis=-1)
        face_img = np.expand_dims(face_img, axis=0)

        # Detect Age
        age = np.round(model_age.predict(face_img / 255.))[0][0]

        # Detect Gender
        gender_arg = np.round(model_gender.predict(face_img / 255.)).astype(np.uint8)
        gender = label_gender[gender_arg[0][0]]

        # Draw
        cv2.putText(img, f'Age: {age}, {gender}', (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (np.random.randint(150, 230), np.random.randint(50, 150), np.random.randint(80, 180)), 1, cv2.LINE_AA)

    # Save annotated image and wait for key press
    cv2.imwrite(f'./image/test/test1.jpg', img)
    cv2.waitKey(0)
    return img

def detect_video(url):
    frame = cv2.VideoCapture(url)
    while True:
        _, img = frame.read()
        img = cv2.flip(img, 1)

        # Load Haar cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_img = cv2.resize(gray[y:y+h, x:x+w], (50, 50))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
            face_img = np.expand_dims(face_img, axis=-1)
            face_img = np.expand_dims(face_img, axis=0)

            # Detect Age
            age = np.round(model_age.predict(face_img / 255.))[0][0]

            # Detect Gender
            gender_arg = np.round(model_gender.predict(face_img / 255.)).astype(np.uint8)
            gender = label_gender[gender_arg[0][0]]

            # Draw
            cv2.putText(img, f'Age: {age}, {gender}', (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (np.random.randint(150, 230), np.random.randint(50, 150), np.random.randint(80, 180)), 1, cv2.LINE_AA)

        cv2.imshow('detect', img)
        if cv2.waitKey(1) == ord('q'):
            break
    frame.release()
    cv2.destroyAllWindows()
