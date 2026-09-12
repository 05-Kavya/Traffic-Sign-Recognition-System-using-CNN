import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# TRAFFIC SIGN CLASSES
# ============================================================

classes = [
    "Speed Limit 20 km/h",
    "Speed Limit 30 km/h",
    "Speed Limit 50 km/h",
    "Speed Limit 60 km/h",
    "Speed Limit 70 km/h",
    "Speed Limit 80 km/h",
    "End of Speed Limit 80 km/h",
    "Speed Limit 100 km/h",
    "Speed Limit 120 km/h",
    "No Passing",
    "No Passing - Trucks",
    "Right of Way",
    "Priority Road",
    "Yield",
    "STOP",
    "No Vehicles",
    "Trucks Prohibited",
    "No Entry",
    "General Caution",
    "Dangerous Curve Left",
    "Dangerous Curve Right",
    "Double Curve",
    "Bumpy Road",
    "Slippery Road",
    "Road Narrows",
    "Road Work",
    "Traffic Signals",
    "Pedestrians",
    "Children Crossing",
    "Bicycles Crossing",
    "Ice/Snow",
    "Wild Animals",
    "End of Speed/Passing Limits",
    "Turn Right Ahead",
    "Turn Left Ahead",
    "Ahead Only",
    "Straight or Right",
    "Straight or Left",
    "Keep Right",
    "Keep Left",
    "Roundabout",
    "End of No Passing",
    "End of No Passing - Trucks"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_traffic_model():
    return load_model("traffic_sign_model.keras")


model = load_traffic_model()


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_traffic_sign(image):

    image = np.array(image)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    image = cv2.resize(
        image,
        (32, 32)
    )

    image = image.astype("float32") / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    prediction = model.predict(
        image,
        verbose=0
    )

    class_id = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    return classes[class_id], confidence


# ============================================================
# LIVE CAMERA PROCESSOR
# ============================================================

class TrafficSignProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # Mirror webcam
        img = cv2.flip(img, 1)

        # Convert to HSV
        hsv = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2HSV
        )

        # Detect red signs
        lower_red1 = np.array([0, 80, 70])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 80, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(
            hsv,
            lower_red1,
            upper_red1
        )

        mask2 = cv2.inRange(
            hsv,
            lower_red2,
            upper_red2
        )

        mask = mask1 + mask2

        # Remove noise
        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        best_box = None
        best_area = 0

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < 500:
                continue

            x, y, w, h = cv2.boundingRect(
                contour
            )

            if w < 30 or h < 30:
                continue

            ratio = w / float(h)

            if 0.5 <= ratio <= 1.8:

                if area > best_area:

                    best_area = area

                    best_box = (
                        x,
                        y,
                        w,
                        h
                    )

        # Classify detected sign
        if best_box is not None:

            x, y, w, h = best_box

            margin = 10

            x1 = max(
                0,
                x - margin
            )

            y1 = max(
                0,
                y - margin
            )

            x2 = min(
                img.shape[1],
                x + w + margin
            )

            y2 = min(
                img.shape[0],
                y + h + margin
            )

            sign = img[
                y1:y2,
                x1:x2
            ]

            if sign.size > 0:

                sign = cv2.resize(
                    sign,
                    (32, 32)
                )

                sign = sign.astype(
                    "float32"
                ) / 255.0

                sign = np.expand_dims(
                    sign,
                    axis=0
                )

                prediction = model.predict(
                    sign,
                    verbose=0
                )

                class_id = np.argmax(
                    prediction
                )

                confidence = (
                    np.max(prediction) * 100
                )

                # Bounding box
                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                # Sign name
                cv2.putText(
                    img,
                    classes[class_id],
                    (x1, max(30, y1 - 35)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                # Confidence
                cv2.putText(
                    img,
                    f"Confidence: {confidence:.1f}%",
                    (x1, max(55, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🚦 Traffic Sign Recognition")

    st.divider()

    st.subheader("Model Information")

    st.write(
        """
        **Model:** CNN

        **Dataset:** GTSRB

        **Classes:** 43

        **Training Images:** 26,684

        **Input Size:** 32 × 32

        **Test Accuracy:** 99.46%
        """
    )

    st.divider()

    st.subheader("Technologies")

    st.write(
        """
        • Python
        • TensorFlow
        • OpenCV
        • NumPy
        • Streamlit
        • Scikit-learn
        """
    )


# ============================================================
# MAIN PAGE
# ============================================================

st.title("🚦 Traffic Sign Recognition System")

st.write(
    "AI-powered traffic sign classification using a "
    "Convolutional Neural Network."
)


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "📷 Upload Image",
        "🎥 Live Camera"
    ]
)


# ============================================================
# IMAGE UPLOAD TAB
# ============================================================

with tab1:

    st.header("Upload a Traffic Sign")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "ppm"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:

            if st.button(
                "🔍 Recognize Traffic Sign",
                use_container_width=True
            ):

                sign_name, confidence = (
                    predict_traffic_sign(image)
                )

                st.success(
                    f"🚦 {sign_name}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.progress(
                    min(int(confidence), 100)
                )


# ============================================================
# LIVE CAMERA TAB
# ============================================================

with tab2:

    st.header("Live Traffic Sign Recognition")

    st.write(
        "Allow camera access and show a red traffic sign "
        "to the camera."
    )

    webrtc_streamer(
        key="traffic-sign-camera",
        video_processor_factory=TrafficSignProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.header("Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Training Images",
        "26,684"
    )

with col2:
    st.metric(
        "Classes",
        "43"
    )

with col3:
    st.metric(
        "Input Size",
        "32 × 32"
    )

with col4:
    st.metric(
        "Test Accuracy",
        "99.46%"
    )


# ============================================================
# ABOUT
# ============================================================

st.divider()

st.header(" About the Project")

st.write(
    """
    This project uses a Convolutional Neural Network (CNN)
    to recognize traffic signs from the GTSRB dataset.

    The system supports both image-based prediction and
    real-time webcam recognition.
    """
)

st.caption(
    "Traffic Sign Recognition System | BSC Data Science OpenCV Project"
)