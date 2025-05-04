# 🎭 Deep Vision a Deepfake Videos Detection Project
This project is a deep learning–based system to detect deepfake videos using a pre-trained Xception model. It offers two interfaces:

- ✅ A Streamlit app for a quick standalone web interface
- ✅ A FastAPI backend with an included web frontend where users can upload a video and see real-time predictions — making it suitable for easy deployment on servers or cloud platforms

<br>

## 📌 Features
- Detect deepfake videos using the Xception neural network
- Streamlit app for video upload and result display
- FastAPI backend with a built-in web frontend for uploading videos and viewing real-time predictions and easy of use
- Real-time frame extraction and analysis
-  Easy-to-use web interface
-  Pretrained models and tokenizers included

<br>

## 🚀 How It Works
- User uploads a video through the Streamlit app or the FastAPI frontend.
- The backend extracts frames from the video.
- Frames are preprocessed and passed through the Xception model.
- The model predicts whether the video is real or a deepfake, along with a confidence score.
- The app or API returns and displays the final prediction result.

<br>

## Project Structure
<pre>
deep_vision/
├── Fast_api/                    
│   └── xception_api.py                         # FastAPI script 
├── Trained_Model/                              # Folder containing the trained Xception model
├── Xception_fine-tune_early_stopping.py        # Script for training (fine tuning) the Xception model
├── app_xception.py                             # Main app (streamlit)
├── requirements.txt                            # List of dependencies to run the project
└── readme.txt                                  # Project Readme
</pre>

<br>

## Running the project 

### To run the streamlit app : <pre>  streamlit run app_xception.py </pre>

### To run the fast api app : <pre> fastapi run xception_api.py </pre>

<br>

## Dataset that i have used : 
- https://github.com/yuezunli/celeb-deepfakeforensics





