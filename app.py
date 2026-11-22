
import streamlit as st
import tensorflow as tf
import numpy as np
import hashlib
from PIL import Image
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_PATH = "models/best_model_v2.keras"
THRESHOLD_PATH = "results/best_threshold_v2.txt"

IMG_SIZE = (32, 32)

# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="wide"
)

# ==========================================
# 3. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #f1f5f9;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    border: 1px solid #93c5fd;
    color: white;
}

[data-testid="stFileUploader"] {
    background: #151e32;
    border-radius: 12px;
    padding: 15px;
}

h2, h3 {
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    return tf.keras.models.load_model(MODEL_PATH)


# ==========================================
# 5. LOAD THRESHOLD
# ==========================================

@st.cache_data
def load_threshold():

    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(THRESHOLD_PATH)

    with open(THRESHOLD_PATH, "r") as file:
        threshold = float(file.read().strip())

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")

    return threshold


try:
    model = load_model()
    threshold = load_threshold()

except Exception as error:
    st.error(f"Model initialization failed: {error}")
    st.stop()


# ==========================================
# 6. HEADER
# ==========================================

st.html("""
<div style="
    text-align:center;
    padding:35px 15px;
    background:#151e32;
    border:1px solid #293752;
    border-radius:16px;
    margin-bottom:30px;
">
    <h1 style="
        font-size:42px;
        font-weight:800;
        color:#60a5fa;
        margin-bottom:8px;
    ">
        AI IMAGE DETECTOR
    </h1>

    <p style="
        color:#94a3b8;
        font-size:17px;
        margin:0;
    ">
        Deep Learning Based Image Classification System
    </p>
</div>
""")


# ==========================================
# 7. SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🔍 AI Detector")

    st.caption("CNN-powered image classification")

    st.divider()

    st.subheader("Model Information")

    st.write("**Architecture:** CNN")
    st.write("**Dataset:** CIFAKE")
    st.write("**Input Size:** 32 × 32")
    st.write("**Optimizer:** Adam")
    st.write(f"**Threshold:** {threshold}")

    st.divider()

    st.subheader("Model Performance")

    st.metric("Test Accuracy", "90.64%")
    st.metric("Balanced Accuracy", "90.64%")

    st.divider()

    st.caption(
        "Developed using Python, TensorFlow and Streamlit."
    )


# ==========================================
# 8. DASHBOARD METRICS
# ==========================================

col1, col2, col3 = st.columns(3)

metric_style = """
<div style="
    background:#151e32;
    border:1px solid #293752;
    border-radius:14px;
    padding:22px;
    text-align:center;
">
    <div style="
        font-size:29px;
        font-weight:800;
        color:#60a5fa;
    ">{value}</div>

    <div style="
        color:#94a3b8;
        font-size:13px;
        margin-top:7px;
    ">{label}</div>
</div>
"""

with col1:
    st.html(metric_style.format(
        value="90.64%",
        label="TEST ACCURACY"
    ))

with col2:
    st.html(metric_style.format(
        value="0.07",
        label="CLASSIFICATION THRESHOLD"
    ))

with col3:
    st.html(metric_style.format(
        value="120K",
        label="TOTAL DATASET IMAGES"
    ))


st.write("")


# ==========================================
# 9. UPLOAD SECTION
# ==========================================

st.subheader("Upload Your Image")

st.caption(
    "Upload a photograph or AI-generated image for classification."
)

uploaded_file = st.file_uploader(
    "Drag and drop an image or browse files",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# 10. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE,
        Image.Resampling.BILINEAR
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    return np.expand_dims(image_array, axis=0)


# ==========================================
# 11. PREDICTION FUNCTION
# ==========================================

def predict_image(image):

    processed_image = preprocess_image(image)

    probability = float(
        model.predict(
            processed_image,
            verbose=0
        )[0][0]
    )

    if not np.isfinite(probability):
        raise ValueError("Invalid model prediction")

    if probability >= threshold:

        result = "REAL"
        score = probability * 100

    else:

        result = "AI-GENERATED"
        score = (1 - probability) * 100

    return result, probability, score


# ==========================================
# 12. DISPLAY IMAGE AND PREDICTION
# ==========================================

if uploaded_file is not None:

    try:

        image_bytes = uploaded_file.getvalue()

        image_id = hashlib.sha256(
            image_bytes
        ).hexdigest()

        image = Image.open(uploaded_file).convert("RGB")

        image_col, result_col = st.columns(
            2,
            gap="large"
        )

        with image_col:

            st.subheader("Image Preview")

            st.image(
                image,
                use_container_width=True
            )

            st.caption(
                f"Dimensions: {image.width} × {image.height}"
            )

        with result_col:

            st.subheader("Image Analysis")

            st.write(
                "Your image is ready for classification."
            )

            detect_button = st.button(
                "🔍 DETECT IMAGE",
                use_container_width=True
            )

            if detect_button:

                with st.spinner("Analyzing image..."):

                    result, probability, score = predict_image(
                        image
                    )

                st.session_state["prediction"] = {
                    "image_id": image_id,
                    "result": result,
                    "probability": probability,
                    "score": score
                }

        # ==================================
        # 13. DISPLAY RESULTS
        # ==================================

        prediction = st.session_state.get("prediction")

        if (
            prediction is not None
            and prediction["image_id"] == image_id
        ):

            result = prediction["result"]
            score = prediction["score"]
            probability = prediction["probability"]

            st.divider()

            st.subheader("Detection Result")

            if result == "REAL":

                st.html("""
                <div style="
                    background:#10352c;
                    border:1px solid #16845e;
                    border-radius:14px;
                    padding:24px;
                    text-align:center;
                ">
                    <h2 style="
                        color:#4ade80;
                        font-size:29px;
                        margin:0;
                    ">
                        ✓ REAL IMAGE
                    </h2>
                    <p style="color:#cbd5e1;">
                        The model classified this image as REAL.
                    </p>
                </div>
                """)

            else:

                st.html("""
                <div style="
                    background:#3a1c28;
                    border:1px solid #b4536b;
                    border-radius:14px;
                    padding:24px;
                    text-align:center;
                ">
                    <h2 style="
                        color:#fb7185;
                        font-size:29px;
                        margin:0;
                    ">
                        ⚠ AI-GENERATED
                    </h2>
                    <p style="color:#cbd5e1;">
                        The model classified this image as AI-GENERATED.
                    </p>
                </div>
                """)

            st.write("")

            st.metric(
                "Model Score for Predicted Class",
                f"{score:.2f}%"
            )

            st.progress(float(score / 100))

            st.caption(
                f"Raw probability of REAL: {probability:.4f}"
            )

            st.info(
                "This is the model's prediction score, not a "
                "calibrated probability of authenticity."
            )

    except Exception as error:

        st.error(f"Image processing failed: {error}")

else:

    st.info("👆 Upload an image to begin detection.")


# ==========================================
# 14. HOW IT WORKS
# ==========================================

st.divider()

with st.expander("How does this system work?"):

    st.markdown("""
    ### CNN-based image classification

    1. User uploads an image.
    2. Image is converted to RGB.
    3. Image is resized to 32 × 32 pixels.
    4. CNN processes the image.
    5. Sigmoid generates a probability for REAL.
    6. Threshold 0.07 determines the classification.

    **Classification rules:**

    - Probability ≥ 0.07 → REAL
    - Probability < 0.07 → AI-GENERATED
    """)


# ==========================================
# 15. DISCLAIMER
# ==========================================

st.warning(
    "Research prototype: This model was trained on CIFAKE "
    "32 × 32 images. Its reliability on arbitrary modern "
    "AI-generated images is not guaranteed."
)


# ==========================================
# 16. FOOTER
# ==========================================

st.html("""
<div style="
    text-align:center;
    border-top:1px solid #293752;
    padding:25px;
    margin-top:35px;
    color:#94a3b8;
">
    <h4 style="color:#60a5fa;">
        AI IMAGE DETECTOR
    </h4>

    <p>
        Developed using Python • TensorFlow • Streamlit
    </p>

    <p>
        Deep Learning Mini Project
    </p>
</div>
""")
