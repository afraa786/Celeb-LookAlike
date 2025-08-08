from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from src.utils.all_utils import read_yaml, create_directory
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st 
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np


# Load configurations and paths
config = read_yaml('config/config.yaml')
params = read_yaml('params.yaml')

artifacts = config['artifacts']
artifacts_dir = artifacts['artifacts_dir']

# upload image dir
upload_image_dir = artifacts['upload_image_dir']
uploadn_path = os.path.join(artifacts_dir, upload_image_dir)

# pickle format data
pickle_format_data_dir = artifacts['pickle_format_data_dir']
img_pickle_file_name = artifacts['img_pickle_file_name']

raw_local_dir_path = os.path.join(artifacts_dir, pickle_format_data_dir)
pickle_file = os.path.join(raw_local_dir_path, img_pickle_file_name)

# feature path
feature_extraction_dir = artifacts['feature_extraction_dir']
extracted_features_name = artifacts['extracted_features_name']

feature_extraction_path = os.path.join(artifacts_dir, feature_extraction_dir)
features_name = os.path.join(feature_extraction_path, extracted_features_name)

# model params
model_name = params['base']['BASE_MODEL']
include_tops = params['base']['include_top']
poolings = params['base']['pooling']


# Load detector and model
detector = MTCNN()
model = VGGFace(model=model_name, include_top=include_tops, 
                input_shape=(224,224,3), pooling=poolings)

# Load stored features and filenames
feature_list = pickle.load(open(features_name,'rb'))
filenames = pickle.load(open(pickle_file,'rb'))


# Function to save uploaded image
def save_uploaded_image(uploaded_image):
    try:
        create_directory(dirs=[uploadn_path])
        with open(os.path.join(uploadn_path, uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False


# Feature extraction from face image
def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    if len(results) == 0:
        return None

    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]

    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image).astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()
    return result


# Recommend most similar face
def recommend(feature_list, features):
    similarity = []
    for i in range(len(feature_list)):
        sim = cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0]
        similarity.append(sim)

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos


# Streamlit interface
st.title('📸 Face Matching with Celebrity')

# Webcam capture via browser
use_camera = st.checkbox("Use Webcam to Capture Photo")

if use_camera:
    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        # Save camera image temporarily
        image = Image.open(camera_image).convert("RGB")
        create_directory([uploadn_path])
        temp_path = os.path.join(uploadn_path, "captured_image.jpg")
        image.save(temp_path)

        # Extract features and find match
        features = extract_features(temp_path, model, detector)

        if features is not None:
            index_pos = recommend(feature_list, features)
            predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))

            col1, col2 = st.columns(2)
            with col1:
                st.header('Captured Image')
                st.image(image)
            with col2:
                st.header("Seems like " + predicted_actor)
                st.image(filenames[index_pos], width=300)
        else:
            st.warning("No face detected. Try again with better lighting.")

# Image file upload
uploaded_image = st.file_uploader('Or upload an image')

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)
        image_path = os.path.join(uploadn_path, uploaded_image.name)

        features = extract_features(image_path, model, detector)
        if features is not None:
            index_pos = recommend(feature_list, features)
            predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))

            col1, col2 = st.columns(2)
            with col1:
                st.header('Uploaded Image')
                st.image(display_image)
            with col2:
                st.header("Seems like " + predicted_actor)
                st.image(filenames[index_pos], width=300)
        else:
            st.warning("No face detected in uploaded image.")
