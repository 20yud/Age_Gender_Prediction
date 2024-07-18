import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer
import av

# Load models
model30_S = load_model('models/model30_S.h5')
model30_M = load_model('models/model30_M.h5')
vgg16_model = load_model('models/VGG16_model.h5')
resnet50_model = load_model('models/ResNet50_model.h5')
model100 = load_model('models/model100.h5')
model30 = load_model('models/model30.h5')

# Load pre-trained face detector model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess image for different models
def preprocess_image(image, model_type, target_size):
    image = cv2.resize(image, target_size)
    if model_type == 'gray_128':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
    return image

# Function to predict age and gender using the models
def predict_age_and_gender(face_img):
    image_30_S = preprocess_image(face_img, 'gray_128', (128, 128))
    image_30_M = preprocess_image(face_img, 'rgb_224', (224, 224))
    image_vgg16 = preprocess_image(face_img, 'rgb_224', (224, 224))
    image_resnet50 = preprocess_image(face_img, 'rgb_224', (224, 224))
    image_100 = preprocess_image(face_img, 'gray_128', (128, 128))
    image_30 = preprocess_image(face_img, 'gray_128', (128, 128))

    pred_30_S = model30_S.predict(image_30_S)
    pred_gender_30_S = round(pred_30_S[0][0][0])
    pred_age_30_S = round(pred_30_S[1][0][0])

    pred_30_M = model30_M.predict(image_30_M)
    pred_gender_30_M = round(pred_30_M[0][0][0])
    pred_age_30_M = round(pred_30_M[1][0][0])

    pred_vgg16 = vgg16_model.predict(image_vgg16)
    pred_gender_vgg16 = round(pred_vgg16[0][0][0])
    pred_age_vgg16 = round(pred_vgg16[1][0][0])

    pred_resnet50 = resnet50_model.predict(image_resnet50)
    pred_gender_resnet50 = round(pred_resnet50[0][0][0])
    pred_age_resnet50 = round(pred_resnet50[1][0][0])

    pred_100 = model100.predict(image_100)
    pred_gender_100 = round(pred_100[0][0][0])
    pred_age_100 = round(pred_100[1][0][0])

    pred_30 = model30.predict(image_30)
    pred_gender_30 = round(pred_30[0][0][0])
    pred_age_30 = round(pred_30[1][0][0])

    return (pred_gender_30_S, pred_age_30_S), (pred_gender_30_M, pred_age_30_M), (pred_gender_vgg16, pred_age_vgg16), (pred_gender_resnet50, pred_age_resnet50), (pred_gender_100, pred_age_100), (pred_gender_30, pred_age_30)

# Streamlit app title and instructions
st.title("Facial Age and Gender Prediction from Images and Webcam")
st.write("Upload an image or use your webcam to predict age and gender for each detected face")

# File uploader component
file_up = st.file_uploader("Upload an image", type=["jpg", "png"])

# Main app logic for image upload
if file_up is not None:
    # Display the uploaded image
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Convert image to OpenCV format
    image_cv = np.array(image.convert('RGB'))
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Detect faces using OpenCV
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    st.write(f"Detected {len(faces)} faces")

    gender_dict = {0: 'Male', 1: 'Female'}

    # Process each detected face
    for (x, y, w, h) in faces:
        face_img = image_cv[y:y+h, x:x+w]
        (pred_gender_30_S, pred_age_30_S), (pred_gender_30_M, pred_age_30_M), (pred_gender_vgg16, pred_age_vgg16), (pred_gender_resnet50, pred_age_resnet50), (pred_gender_100, pred_age_100), (pred_gender_30, pred_age_30) = predict_age_and_gender(face_img)

        st.write(f"Face at [{x}, {y}, {w}, {h}]:")
        st.write(f"Model 30_S - Predicted Gender: {gender_dict[pred_gender_30_S]}, Predicted Age: {pred_age_30_S}")
        st.write(f"Model 30_M - Predicted Gender: {gender_dict[pred_gender_30_M]}, Predicted Age: {pred_age_30_M}")
        st.write(f"VGG16 model - Predicted Gender: {gender_dict[pred_gender_vgg16]}, Predicted Age: {pred_age_vgg16}")
        st.write(f"ResNet50 model - Predicted Gender: {gender_dict[pred_gender_resnet50]}, Predicted Age: {pred_age_resnet50}")
        st.write(f"Model 100 - Predicted Gender: {gender_dict[pred_gender_100]}, Predicted Age: {pred_age_100}")
        st.write(f"Model 30 - Predicted Gender: {gender_dict[pred_gender_30]}, Predicted Age: {pred_age_30}")

        # Draw rectangle around the face
        cv2.rectangle(image_cv, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display processed image with detected faces
    st.image(image_cv, caption='Processed Image with Detected Faces.', use_column_width=True)

# Function to handle video frames from webcam
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Resize image to reduce lag (240x120)
    img_resized = cv2.resize(img, (540, 360))

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    gender_dict = {0: 'Male', 1: 'Female'}

    for (x, y, w, h) in faces:
        face_img = img_resized[y:y+h, x:x+w]
        (pred_gender_30_S, pred_age_30_S), _, _, _, (pred_gender_100, pred_age_100), (pred_gender_30, pred_age_30) = predict_age_and_gender(face_img)
        
        # Display predictions on the video stream
        gender_text = f"Gender: {gender_dict[pred_gender_100]}"
        age_text = f"Age: {pred_age_100}"
        cv2.rectangle(img_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_resized, gender_text, (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img_resized, age_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img_resized, format="bgr24")

# Streamlit app title for webcam section
st.title('Webcam - Facial Age and Gender Prediction')

# Initialize WebRTC streamer with the video_frame_callback
webrtc_ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback)

# Display the webcam streamer
if webrtc_ctx.video_transformer:
    if not webrtc_ctx.state.playing:
        st.write('Waiting for webcam to start...')
