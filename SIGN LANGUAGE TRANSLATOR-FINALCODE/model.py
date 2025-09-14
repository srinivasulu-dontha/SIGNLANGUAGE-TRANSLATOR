import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

BASE_SAVE_DIR = Path(r"C:\Users\USER\Desktop\sign launguage translator\captured_images")
MODEL_PATH = r"C:\Users\USER\Desktop\sign launguage translator\sign_model.pkl"

mp_hands = mp.solutions.hands

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
        print("⚠️ No valid data found! Training aborted.")
        return None

    df = pd.DataFrame(data)
    df['label'] = labels

    X = df.drop('label', axis=1)
    y = df['label']

    # Train-test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model accuracy: {acc:.2f}")

    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    print(f"💾 Model saved to {MODEL_PATH}")

    return clf, acc

if __name__ == "__main__":
    train_model()
