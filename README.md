# Face Matching Application

A **Streamlit based** web application that matches an uploaded face image with the most similar celebrity face from a dataset using **deep learning-based facial feature extraction**.

<img width="1600" height="900" alt="Face Matching App Screenshot" src="https://github.com/user-attachments/assets/c6c6df9f-f223-4286-bc1d-15f1a711ae1b" />

---

## 📌 Project Overview
This application allows users to upload an image of a face, then finds and displays the **most similar celebrity** from a preloaded dataset.

**Technologies Used:**
- **VGGFace2 ResNet-50** for feature extraction
- **Cosine Similarity** for matching
- **MTCNN** for face detection
- **Streamlit** for the web interface
- **Kaggle Celebrity Dataset**

**Use Cases:**
- Fun celebrity look-alike finder
- Educational tool for facial recognition concepts
- Demonstration of transfer learning in computer vision

---

## 📂 Repository Structure
├── artifacts/
│ ├── extracted_features/ 
│ ├── pickle_format_data/
│ └── upload/
├── config/
│
├── data/ 
│
├── logs/ 
│
├── src/ 
│ ├── utils/ 
│ └── pycache/
│
├── run.py 
├── requirements.txt 
└── README.md 

---

## ⚙️ How It Works

1. **Dataset Preparation**
   - The Kaggle celebrity dataset is preprocessed:
     - Faces are cropped and aligned
     - Embeddings generated using **ResNet-50 (VGGFace2)**
     - Features stored in `.pkl` files for fast search

2. **Uploading an Image**
   - User uploads a face image via Streamlit UI

3. **Feature Extraction**
   - Face detected and cropped using **MTCNN**
   - Features extracted using **VGGFace2 ResNet-50**

4. **Similarity Matching**
   - **Cosine similarity** calculated between uploaded image and dataset embeddings
   - Top match (or top-N matches) retrieved

5. **Results Display**
   - Celebrity’s image, name, and similarity score shown

---

## 🖥️ Installation & Running the Application

### Step 1: Clone the Repository
```bash
git clone https://github.com/afraa786/face-matching-application.git
cd face-matching-application
```

### Step 2: Create a Virtual Environment 
Using Conda
```bash
conda create --name face_match_env python=3.8
conda activate face_match_env
```

### Step 3: Install Required Packages
Using conda:
```bash
conda install --file requirements.txt
```
## 🔍 Algorithms Used

- **MTCNN (Multi-task Cascaded Convolutional Networks):**  
  Used for accurate face detection and alignment on the input images before feature extraction.

- **ResNet-50 (VGGFace2 Model):**  
  A deep convolutional neural network pretrained on the VGGFace2 dataset to extract robust facial embeddings (feature vectors) from images.

- **Cosine Similarity:**  
  Calculates the similarity between the feature vector of the uploaded face and those of celebrities to find the closest match.

- **Face Embedding Extraction:**  
  Converts face images into fixed-length numerical vectors capturing distinctive facial features for comparison.

## 📊 Example Output

When an image is uploaded, the app:

- Shows the uploaded image
- Displays the closest celebrity match with similarity score
- Optionally lists top-N closest matches


## 🙌 Acknowledgements
- [VGGFace2 Dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) – Large-scale face dataset for training deep face recognition models.
- [Kaggle Celebrity Dataset](https://www.kaggle.com/datasets/sroy93/bollywood-celeb-localized-face-dataset-extended) – Bollywood celebrity face dataset used for matching.
- [Streamlit](https://streamlit.io/) – Open-source app framework for Machine Learning and Data Science.


