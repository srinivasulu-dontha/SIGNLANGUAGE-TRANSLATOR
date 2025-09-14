import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np
import av
import pickle
from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================
# Paths & MediaPipe Setup
# ==========================
BASE_SAVE_DIR = Path(r"C:\Users\USER\Desktop\sign launguage translator\captured_images")
MODEL_PATH = r"C:\Users\USER\Desktop\sign launguage translator\sign_model.pkl"
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
mp_hands = mp.solutions.hands

# Load trained model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

# ==========================
# Helper Functions
# ==========================
def get_label_dir(label):
    label_dir = BASE_SAVE_DIR / label
    label_dir.mkdir(exist_ok=True)
    return label_dir

def save_image(img, label, count):
    label_dir = get_label_dir(label)
    filepath = label_dir / f"{label}_{count}.jpg"
    cv2.imwrite(str(filepath), img)
    return filepath

# Extract landmarks (for training)
def extract_landmarks(image_path, hands):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None
    landmarks = []
    for lm in result.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

# Training function
def train_model():
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
    data, labels = [], []

    for label_dir in BASE_SAVE_DIR.iterdir():
        if label_dir.is_dir():
            label = label_dir.name
            for img_path in label_dir.glob("*.jpg"):
                features = extract_landmarks(img_path, hands)
                if features:
                    data.append(features)
                    labels.append(label)

    hands.close()

    if not data:
        return None, None

    df = pd.DataFrame(data)
    df['label'] = labels

    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)

    return clf, acc

# ==========================
# Video Processor - Capture Mode
# ==========================
class CaptureProcessor:
    def __init__(self):
        self.save_next = False
        self.last_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.last_frame = img

        if self.save_next:
            self._save_frame(img)
            self.save_next = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _save_frame(self, img):
        label = st.session_state.get("sign_label", "")
        if label:
            if label not in st.session_state.image_counts:
                st.session_state.image_counts[label] = 0
            st.session_state.image_counts[label] += 1
            count = st.session_state.image_counts[label]
            save_image(img, label, count)
            st.sidebar.success(f"Saved frame '{label}_{count}.jpg'")
        else:
            st.sidebar.warning("Enter a sign label first")

# ==========================
# Video Processor - Recognition Mode
# ==========================
class RecognitionProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if model:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self.hands.process(img_rgb)
            if result.multi_hand_landmarks:
                landmarks = []
                for lm in result.multi_hand_landmarks[0].landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                input_data = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(input_data)[0]
                cv2.putText(img, f"Sign: {prediction}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, "No hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Model not found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================
# Streamlit UI
# ==========================
st.title("✋ Sign Language Recognition App")

if "image_counts" not in st.session_state:
    st.session_state.image_counts = {}
if "capture_on" not in st.session_state:
    st.session_state.capture_on = False
if "live_on" not in st.session_state:
    st.session_state.live_on = False
if "capture_processor" not in st.session_state:
    st.session_state.capture_processor = CaptureProcessor()

sign_label = st.sidebar.text_input("Enter Sign Label (e.g. HELLO)").upper().strip()
st.session_state.sign_label = sign_label
capture_processor = st.session_state.capture_processor

# ==========================
# Capture Mode
# ==========================
st.sidebar.subheader("Capture Mode")
if st.sidebar.button("Start Capture Mode"):
    st.session_state.capture_on = True
if st.sidebar.button("Stop Capture Mode"):
    st.session_state.capture_on = False

if st.session_state.capture_on:
    st.subheader("📸 Capture Mode (Live Preview)")
    webrtc_streamer(
        key="sign-capture",
        video_processor_factory=lambda: capture_processor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    if st.sidebar.button("Capture Frame Now"):
        if capture_processor.last_frame is not None:
            capture_processor._save_frame(capture_processor.last_frame)
        else:
            st.sidebar.warning("No frame available yet!")

# ==========================
# Show Collected Data
# ==========================
if st.sidebar.button("Show Collected Data"):
    st.subheader("📂 Collected Data Summary")
    total_images = 0
    for label_dir in sorted(BASE_SAVE_DIR.iterdir()):
        if label_dir.is_dir():
            images = list(label_dir.glob("*.jpg"))
            count = len(images)
            total_images += count
            st.markdown(f"### 🏷️ Label: **{label_dir.name}** — {count} images")
            if images:
                cols = st.columns(min(5, len(images)))
                for i, img_path in enumerate(images[:5]):
                    with cols[i % 5]:
                        st.image(str(img_path), caption=f"{img_path.name}", use_container_width=True)
    if total_images == 0:
        st.info("No collected images found")

# ==========================
# Delete Labels
# ==========================
existing_labels = [d.name for d in BASE_SAVE_DIR.iterdir() if d.is_dir()]
if existing_labels:
    delete_choice = st.sidebar.selectbox("Select Label to Delete", [""] + existing_labels)
    if delete_choice and st.sidebar.button("Delete Selected Label"):
        shutil.rmtree(BASE_SAVE_DIR / delete_choice)
        st.sidebar.success(f"Deleted all images for label '{delete_choice}'")
else:
    st.sidebar.info("No labels available to delete")

# ==========================
# Train Model
# ==========================
st.sidebar.subheader("Model Training")
if st.sidebar.button("Train / Update Model"):
    clf, acc = train_model()
    if clf:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        st.sidebar.success(f"✅ Model retrained (accuracy: {acc:.2f})")
    else:
        st.sidebar.error("⚠️ Training failed. No data found.")

# ==========================
# Live Recognition
# ==========================
st.sidebar.subheader("Live Recognition")
if st.sidebar.button("Start Live Recognition"):
    if model is None:
        st.sidebar.error("⚠️ No trained model found. Please train the model first.")
    else:
        st.session_state.live_on = True

if st.sidebar.button("Stop Live Recognition"):
    st.session_state.live_on = False

if st.session_state.live_on:
    st.subheader("🔴 Live Sign Recognition")
    webrtc_streamer(
        key="sign-recognition",
        video_processor_factory=RecognitionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
else:
    st.info("👉 Use **Start Capture Mode** to collect images, or **Train Model** then **Start Live Recognition**.")
