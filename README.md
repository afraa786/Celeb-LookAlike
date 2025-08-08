# Face Matching Application

Streamlit-based web application to match the uploaded face with a celebrity face.
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/c6c6df9f-f223-4286-bc1d-15f1a711ae1b" />


## Project Description
- A web application using streamlit to match the uploaded face with a celebrity face
- The pipeline uses the pre trained ResNet 50 VGGFace 2 model to extract features from the uploaded image and uses cosine distance to find the most similar celebrity face with the uploaded image
- The dataset is from Kaggle

## Files and data description

<pre>
├───artifacts
│   ├───extracted_features
│   ├───pickle_format_data
│   └───upload
├───config
├───data
├───logs
├───src
│   ├───utils
│   │   └───__pycache__
│   └───__pycache__
└───src.egg-info

</pre>

## Running Files
<pre>
Step 1: Create Virtual Environment:
    - conda create --name face_match_env python=3.8
    - conda activate face_match_env
Step 2 : Install packages
    - conda install --file requirements.txt
Step 3 : Run the Application
    - python run.py
</pre>
