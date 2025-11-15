import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw

# ------------------------------
# PAGE SETUP AND CSS
# ------------------------------
st.set_page_config(page_title="ASL Alphabet Detector", layout="wide")
st.markdown("""
<style>
body { background-color: #f0f2f6; }
.stApp { font-family: 'Segoe UI', sans-serif; }
.title { text-align: center; font-size: 45px; font-weight: bold; color: #4CAF50; margin-bottom:10px; }
.subtitle { text-align: center; font-size: 18px; color: #555; margin-bottom: 40px; }
.prediction-card { background-color: #ffffff; padding: 20px; border-radius: 15px; 
                  box-shadow: 2px 2px 12px rgba(0,0,0,0.15); margin-bottom: 20px; text-align:center;}
.prediction-text { font-size: 28px; font-weight: bold; color: #333; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🖐️ ASL Alphabet Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Show a hand sign in front of the camera or upload an image to detect the ASL letter.</div>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('asl_cnn_model_normalized.keras', compile=False)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None
model = load_model()
IMG_SIZE = (224, 224)

# ------------------------------
# CLASS LABELS
# ------------------------------
class_labels = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R',
    'S','T','U','V','W','X','Y','Z','space','del','nothing'
]

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def preprocess_image(img_pil):
    """Resize and convert image for model prediction"""
    img_resized = img_pil.resize(IMG_SIZE)
    img_array = np.array(img_resized).astype("float32")
    if img_array.ndim == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    return np.expand_dims(img_array, axis=0)

def predict_asl(img_pil):
    """Predict top-5 ASL classes"""
    img_array = preprocess_image(img_pil)
    preds = model.predict(img_array, verbose=0)[0]
    top5_idx = np.argsort(preds)[::-1][:5]
    return [(class_labels[i], float(preds[i]*100)) for i in top5_idx]

def resize_for_display(img_pil, max_size=400):
    w, h = img_pil.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        return img_pil.resize((int(w*scale), int(h*scale)))
    return img_pil

# ------------------------------
# CAMERA INPUT + UPLOAD
# ------------------------------
st.subheader("📷 Capture Image / Upload Image")
col1, col2 = st.columns(2)

with col1:
    img_camera = st.camera_input("Show hand sign in front of camera")

with col2:
    img_upload = st.file_uploader("Or upload an image", type=["jpg","jpeg","png"])

# Select input image
input_img = None
if img_camera is not None:
    input_img = Image.open(img_camera).convert("RGB")
elif img_upload is not None:
    input_img = Image.open(img_upload).convert("RGB")

# ------------------------------
# DISPLAY AND PREDICTION
# ------------------------------
if input_img is not None and model is not None:
    # Draw ROI rectangle
    img_draw = resize_for_display(input_img.copy())
    w, h = img_draw.size
    draw = ImageDraw.Draw(img_draw)
    x1, y1 = int(w*0.05), int(h*0.05)
    x2, y2 = int(w*0.95), int(h*0.95)
    draw.rectangle([x1, y1, x2, y2], outline="green", width=4)
    st.image(img_draw, caption="Input Image (ROI in green)", use_column_width=True)

    # Predict
    top5 = predict_asl(input_img)
    st.markdown(f"<div class='prediction-card'><span class='prediction-text'>Top prediction: {top5[0][0]} ({top5[0][1]:.2f}%)</span></div>", unsafe_allow_html=True)

    st.markdown("**Top-5 Predictions:**")
    for label, conf in top5:
        st.progress(conf/100.0)
        st.write(f"{label}: {conf:.2f}%")

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Developed with ❤️ using Streamlit, TensorFlow, and PIL</p>", unsafe_allow_html=True)