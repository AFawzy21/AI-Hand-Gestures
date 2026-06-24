# 🎮 Gesture Order System

A hands-free ordering app driven by **hand gestures** from your webcam — with full **mouse** support so it works the same with or without a camera. Built with Streamlit, MediaPipe, and a small SVM classifier.

Show a gesture (or click the matching button) to browse menus, pick items, build a cart, and confirm an order.

---

## ✋ Gestures

| Gesture | Icon | Action  | Meaning                          |
| ------- | :--: | ------- | -------------------------------- |
| Next     | 👍 | Next    | Move to the next option          |
| Previous | 👎 | Previous| Move to the previous option      |
| Accept   | ☝️ | Select  | Confirm / add / remove / proceed |
| Back     | 🖐️ | Back    | Go up one level                  |
| Idle     | ✊ | —       | No action                        |

Every gesture also appears as a labeled button under the camera, so the whole flow is usable with the mouse alone.

---

## 🧩 How it works

The app is three files working together:

```
app.py                    ← Streamlit UI + camera loop + ordering logic
Model/MediapipeModel.py   ← hand detection (MediaPipe) → 21 landmarks
Model/SvmModel.py         ← landmarks → gesture label (SVM)
```

**Pipeline per frame:**

1. A background thread in `app.py` reads webcam frames with OpenCV.
2. `MediapipeModel.get_landmarks(frame)` returns the 21 hand landmarks (or `None`).
3. `MediapipeModel.flatten_landmarks(...)` flattens them to a 63-value vector (`x, y, z` × 21).
4. `SvmModel.predict_svc(vector)` returns one of `Accept`, `Back`, `Idle`, `Next`, `Previous`.
5. `app.py` turns that label into a UI action and updates the screen live.

### `Model/MediapipeModel.py`
Loads MediaPipe's `gesture_recognizer.task` and exposes:
- **`get_landmarks(image)`** — runs hand detection on an RGB image; returns the detected hand landmarks or `None`.
- **`flatten_landmarks(landmarks)`** — flattens one hand's landmarks into a flat list `[x, y, z, ...]` for the classifier.

### `Model/SvmModel.py`
Loads the trained `svc_model.joblib` and exposes:
- **`predict_svc(landmarks)`** — takes a flattened landmark vector and returns a gesture label from `['Accept', 'Back', 'Idle', 'Next', 'Previous']`.

### `app.py`
The Streamlit front end and the application logic:
- **Background camera worker** — owns the webcam in its own thread, so detection never blocks the UI. Reopens the camera on failure and (on Windows) prefers the quieter DirectShow backend.
- **In-place rendering** — the video feed, predicted gesture, info panel, and buttons update in place, so showing a gesture does **not** freeze or flicker the feed.
- **Mouse + gesture parity** — both routes call the same `process_gesture()` handler.
- **Live prediction readout** — the current model output (including `Idle` / `No hand`) is shown under the camera.
- **Ordering flow** — Main menu → pick category → pick item → cart → summary → reorder.

---

## 📂 Project structure

```
.
├── app.py                       # main Streamlit app (run this)
└── Model/
    ├── MediapipeModel.py        # hand landmark detection
    ├── SvmModel.py              # gesture classification
    ├── gesture_recognizer.task  # MediaPipe model (required at runtime)
    └── svc_model.joblib         # trained SVM model (required at runtime)
```

> `gesture_recognizer.task` and `svc_model.joblib` are loaded at import time and must be present in the `Model/` folder.

---

## 🚀 Getting started

### Requirements
- Python 3.10+
- A webcam (only needed for gesture control — the app is fully usable with the mouse without one)

### Install

```bash
pip install -r requirements.txt
```

Or install the packages directly:

```bash
pip install streamlit opencv-python mediapipe scikit-learn joblib numpy
```

### Dependencies

| Package         | Used by                              | Purpose                                  |
| --------------- | ------------------------------------ | ---------------------------------------- |
| `streamlit`     | `app.py`                             | Web UI and app loop                      |
| `opencv-python` | `app.py`, `Model/MediapipeModel.py`  | Webcam capture and image handling        |
| `mediapipe`     | `Model/MediapipeModel.py`            | Hand detection and landmark extraction   |
| `scikit-learn`  | `Model/SvmModel.py`                  | SVM model (loaded from `svc_model.joblib`)|
| `joblib`        | `Model/SvmModel.py`                  | Loading the saved model                  |
| `numpy`         | `Model/MediapipeModel.py`            | Array handling                           |

### Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually <http://localhost:8501>).

---

## 🕹️ Using the app

1. Use the on-screen **buttons** to navigate — this works immediately, no camera needed.
2. To control with gestures, flip **“Enable camera & gestures”** in the sidebar.
3. Watch the **Predicted gesture** line under the feed to see what the model sees.
4. Build your cart, open **Order Summary**, confirm, then start a new order or keep shopping.

### Ordering flow

```
Main Menu ──Buy──▶ Select Category ──▶ Select Item ──▶ (adds to Cart)
   │                                                        │
   ├── Remove from Stack ──▶ remove items from the cart     │
   ├── Accept Order ──▶ Order Summary ──▶ Order Complete ───┘
   └── Cancel Order ──▶ clears the cart
```

---

## 🔧 Tuning

- **Detection rate** — `DETECTION_INTERVAL` in `app.py` (seconds between predictions; default `1.0`). Lower = more responsive, higher CPU.
- **Camera index / backend** — change `cv2.VideoCapture(0, cv2.CAP_DSHOW)` in `app.py` if you have multiple cameras or are not on Windows.
- **Menu contents** — edit the `CATEGORIES`, `MAIN_MENU`, and `REORDER_MENU` constants in `app.py`.

---

## 🛠️ Troubleshooting

| Problem | Fix |
| ------- | --- |
| “Cannot access camera” / black feed | Close other apps using the webcam; try a different camera index in `cv2.VideoCapture`. |
| Gestures not recognized | Improve lighting, keep your whole hand in frame, hold the pose ~1 second. |
| `FileNotFoundError` on startup | Ensure `gesture_recognizer.task` and `svc_model.joblib` exist in `Model/`. |
| Noisy OpenCV `grabFrame` warnings | Already suppressed in `app.py`; the DirectShow backend avoids most of them on Windows. |
