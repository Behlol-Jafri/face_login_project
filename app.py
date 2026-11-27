import pickle
import re
import sqlite3
import urllib.request
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

DB_FILE = "face_users.db"
MODEL_DIR = Path("models")
ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
DETECTOR = "face_detection_yunet/face_detection_yunet_2023mar.onnx"
RECOGNIZER = "face_recognition_sface/face_recognition_sface_2021dec.onnx"
MATCH_THRESHOLD = 0.45  # higher = stricter (OpenCV's suggested minimum is 0.363)

# ---------- Your sentiment model (change these to match your training) ----------
SENTIMENT_MODEL = "sentiment_model.keras"  # or "sentiment_model.h5"
TOKENIZER_FILE = "tokenizer.pkl"  # set to None if the model accepts raw text
MAX_LEN = 50        # same maxlen as in training
PADDING = "post"      # "pre" or "post", same as in training
TRUNCATING = "pre"   # "pre" or "post", same as in training
LABELS = ["Irrelevant", "Negative", "Neutral", "Positive"]  # same order as in training (0, 1, 2, 3)

ANGLES = [
    "Look straight at the camera",
    "Turn your face slightly LEFT",
    "Turn your face slightly RIGHT",
    "Tilt your face slightly UP",
    "Tilt your face slightly DOWN",
]


# ---------- Face recognition ----------
@st.cache_resource(show_spinner="Loading face models (first run downloads them)...")
def load_models():
    MODEL_DIR.mkdir(exist_ok=True)
    paths = []
    for name in (DETECTOR, RECOGNIZER):
        path = MODEL_DIR / Path(name).name
        if not path.exists():
            temp = path.with_suffix(".part")
            urllib.request.urlretrieve(ZOO + name, temp)
            temp.rename(path)
        paths.append(str(path))
    detector = cv2.FaceDetectorYN.create(paths[0], "", (320, 320), 0.8)
    recognizer = cv2.FaceRecognizerSF.create(paths[1], "")
    return detector, recognizer


def get_embedding(image_bytes):
    """Return (embedding, error_message) for a photo with exactly one face."""
    detector, recognizer = load_models()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None, "Could not read the photo. Please try again."
    detector.setInputSize((image.shape[1], image.shape[0]))
    _, faces = detector.detect(image)
    if faces is None:
        return None, "No face detected. Improve the lighting and try again."
    if len(faces) > 1:
        return None, "More than one face detected. Only you should be in the frame."
    feature = recognizer.feature(recognizer.alignCrop(image, faces[0]))
    feature = feature.flatten().astype(np.float32)
    return feature / np.linalg.norm(feature), None


# ---------- Database ----------
def connect():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with closing(connect()) as conn, conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, "
            "email TEXT UNIQUE, age INTEGER, gender TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS photos (user_id INTEGER REFERENCES users(id), "
            "angle TEXT, photo BLOB, embedding BLOB)"
        )


def email_exists(email):
    with closing(connect()) as conn:
        row = conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone()
    return row is not None


def create_user(name, email, age, gender, captures):
    try:
        with closing(connect()) as conn, conn:  # one transaction
            user_id = conn.execute(
                "INSERT INTO users (name, email, age, gender) VALUES (?, ?, ?, ?)",
                (name, email, age, gender),
            ).lastrowid
            conn.executemany(
                "INSERT INTO photos VALUES (?, ?, ?, ?)",
                [(user_id, angle, photo, emb.tobytes())
                 for angle, (photo, emb) in zip(ANGLES, captures)],
            )
        return True
    except sqlite3.IntegrityError:
        return False


def find_match(embedding):
    """Compare with every stored photo. Return (user, score)."""
    with closing(connect()) as conn:
        rows = conn.execute(
            "SELECT u.name, u.email, u.age, u.gender, p.embedding "
            "FROM users u JOIN photos p ON p.user_id = u.id"
        ).fetchall()
    best_user, best_score = None, -1.0
    for name, email, age, gender, blob in rows:
        score = float(np.dot(embedding, np.frombuffer(blob, np.float32)))
        if score > best_score:
            best_user = {"name": name, "email": email, "age": age, "gender": gender}
            best_score = score
    if best_score < MATCH_THRESHOLD:
        return None, best_score
    return best_user, best_score


# ---------- Sentiment analysis ----------
@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model():
    import tensorflow as tf

    model = tf.keras.models.load_model(SENTIMENT_MODEL, compile=False)
    tokenizer = None
    if TOKENIZER_FILE:
        with open(TOKENIZER_FILE, "rb") as file:
            tokenizer = pickle.load(file)  # only load a file you created yourself
    return model, tokenizer


def predict_sentiment(text):
    import tensorflow as tf

    model, tokenizer = load_sentiment_model()
    # If you cleaned the text during training, apply the same cleaning here.
    if tokenizer is None:
        inputs = tf.constant([text])
    else:
        sequences = tokenizer.texts_to_sequences([text])
        if not sequences[0]:
            return "Not sure (no known words in this message)"
        inputs = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=MAX_LEN, padding=PADDING, truncating=TRUNCATING
        )
    scores = model.predict(inputs, verbose=0)[0]
    if len(scores) != len(LABELS):
        raise ValueError(f"Model has {len(scores)} outputs, but LABELS has {len(LABELS)}.")
    return LABELS[int(np.argmax(scores))]


# ---------- Navigation ----------
def go_to(page):
    st.session_state.page = page
    st.session_state.captures = []  # discard unfinished photos
    st.session_state.attempt += 1   # reset the camera widgets


def logout():
    st.session_state.user = None
    st.session_state.messages = []
    go_to("home")


# ---------- Pages ----------
def home_page():
    st.title("Face Login System")
    st.write("If you have already signed up, please log in. Otherwise, sign up.")
    col1, col2 = st.columns(2)
    col1.button("Login", on_click=go_to, args=("login",), use_container_width=True)
    col2.button("Sign Up", on_click=go_to, args=("signup",), use_container_width=True)


def signup_page():
    st.title("Sign Up")
    name = st.text_input("Name").strip()
    email = st.text_input("Email").strip().lower()
    age = st.number_input("Age", min_value=5, max_value=120, value=None, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=None)

    st.subheader("Face Photos (5 angles)")
    captures = st.session_state.captures
    step = len(captures)
    st.progress(step / len(ANGLES), text=f"{step} of {len(ANGLES)} photos captured")

    if step < len(ANGLES):
        st.info(f"Photo {step + 1}: {ANGLES[step]}")
        shot = st.camera_input(
            "Take photo", key=f"cam_{st.session_state.attempt}_{step}"
        )
        if shot:
            data = shot.getvalue()
            embedding, error = get_embedding(data)
            if error:
                st.error(error + " Click 'Clear photo' to retake.")
            elif captures and np.dot(captures[0][1], embedding) < 0.3:
                st.error("This face does not match your first photo. Please retake.")
            else:
                captures.append((data, embedding))
                st.rerun()
    else:
        st.success("All 5 photos captured.")
        st.image([photo for photo, _ in captures], width=120)

    if step:
        st.button("Retake all photos", on_click=go_to, args=("signup",))

    if st.button("Create Account", type="primary", disabled=step < len(ANGLES)):
        if not name or not email or age is None or gender is None:
            st.error("Please fill in all fields.")
        elif not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            st.error("Please enter a valid email.")
        elif email_exists(email) or find_match(captures[0][1])[0]:
            st.warning("You are already signed up. Please log in.")
        elif create_user(name, email, int(age), gender, captures):
            # st.success("Account created successfully! You can now log in.")
            go_to("login")
        else:
            st.warning("You are already signed up. Please log in.")

    # st.button("Go to Login", on_click=go_to, args=("login",))
    st.button("Back", on_click=go_to, args=("home",))


def login_page():
    st.title("Face Login")
    st.write("Look straight at the camera and take one photo.")
    shot = st.camera_input("Take photo", key=f"login_{st.session_state.attempt}")

    if shot:
        embedding, error = get_embedding(shot.getvalue())
        if error:
            st.error(error)
        else:
            user, _ = find_match(embedding)
            if user:
                st.session_state.user = user
                go_to("dashboard")
                st.rerun()
            st.error("Face not recognized. Try again, or sign up if you are new.")

    # st.button("Go to Sign Up", on_click=go_to, args=("signup",))
    st.button("Back", on_click=go_to, args=("home",))


def dashboard_header(user):
    with st.container(border=True):
        left, right = st.columns([4, 1], vertical_alignment="center")
        with left:
            st.subheader(user["name"])
            st.write(f"Age: {user['age']}  |  Gender: {user['gender']}")
        right.button("Logout", on_click=logout, use_container_width=True)


def show_message(sender, text):
    with st.container(border=True):
        st.markdown(f"**{sender}**")
        st.text(text)


def chatbot():
    st.subheader("Sentiment Chatbot")
    missing = [f for f in (SENTIMENT_MODEL, TOKENIZER_FILE) if f and not Path(f).exists()]
    if missing:
        st.warning("Put these files in the app folder: " + ", ".join(missing))
        return

    for sender, text in st.session_state.messages:
        show_message(sender, text)

    message = st.chat_input("Type your message...")
    if message and message.strip():
        try:
            reply = predict_sentiment(message)
        except Exception as error:
            st.error(f"Model error: {error}")
            return
        for sender, text in (("You", message), ("Model", reply)):
            st.session_state.messages.append((sender, text))
            show_message(sender, text)


def dashboard_page():
    user = st.session_state.user
    if not user:
        go_to("home")
        st.rerun()
    dashboard_header(user)
    chatbot()


# ---------- Main ----------
st.set_page_config(page_title="Face Login System",)
init_db()

st.session_state.setdefault("page", "home")
st.session_state.setdefault("user", None)
st.session_state.setdefault("captures", [])
st.session_state.setdefault("attempt", 0)
st.session_state.setdefault("messages", [])

PAGES = {
    "home": home_page,
    "signup": signup_page,
    "login": login_page,
    "dashboard": dashboard_page,
}
PAGES[st.session_state.page]()