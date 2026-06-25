import os

# Silence MSMF's noisy "can't grab frame" warnings and prefer the more
# stable DirectShow backend on Windows. Must be set before importing cv2.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

import streamlit as st
import cv2
import time
import threading

# Extra safety: mute OpenCV's internal logger if the API is available.
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

from Model.MediapipeModel import get_landmarks, flatten_landmarks
from Model.SvmModel import predict_svc

# ============================================================
#  Page config & global styles
# ============================================================
st.set_page_config(page_title="Gesture Order System", page_icon="🎮", layout="wide")

st.markdown(
    """
    <style>
        /* Collapse Streamlit's default top/bottom padding so everything fits one screen */
        .block-container { padding-top: 1.2rem !important; padding-bottom: 0.5rem !important; max-width: 100% !important; }
        header[data-testid="stHeader"] { height: 0; }
        #MainMenu, footer { visibility: hidden; }

        .stButton > button {
            width: 100%;
            height: 56px;
            font-size: 17px;
            font-weight: 600;
            border-radius: 12px;
        }
        .gesture-badge {
            display:inline-block; padding:2px 10px; margin:2px;
            border-radius:20px; background:#262730; font-size:13px;
        }
        /* The menu card sits in the centre of the D-pad cross. A glowing
           border makes the correlation "this is what the buttons act on" obvious. */
        .stage-card {
            background:linear-gradient(160deg,#23232e,#1a1a22);
            border:1px solid #3a3a52; border-radius:16px;
            padding:18px 20px; text-align:center;
            box-shadow:0 0 0 1px #2a2a3a, 0 8px 24px rgba(120,120,255,.12);
        }
        .stage-title { color:#9aa0ff; font-size:13px; letter-spacing:2px; }
        .stage-main  { font-size:30px; font-weight:800; margin:6px 0; line-height:1.15; }
        .stage-hint  { color:#888; font-size:14px; }
        /* In the D-pad, the side (Prev/Next) buttons are tall so they hug the
           full height of the menu card and read as "left / right". The keyed
           containers below emit stable `st-key-…` classes we can target. */
        div[class*="st-key-dpad-prev"] .stButton > button,
        div[class*="st-key-dpad-next"] .stButton > button { height: 132px; }
        /* Tall dashed placeholder for an unavailable side direction. */
        div[class*="st-key-dpad-prev"] .dpad-ghost,
        div[class*="st-key-dpad-next"] .dpad-ghost { height: 132px; }
        /* Dim slots where a direction isn't available for this stage, so the
           cross keeps its shape and the user learns which gestures do nothing. */
        .dpad-ghost {
            height:56px; border:1px dashed #333; border-radius:12px;
            display:flex; align-items:center; justify-content:center;
            color:#555; font-size:22px;
        }
        .dpad-ghost.tall { height:132px; }
        /* Keep the camera image from growing taller than the viewport */
        div[data-testid="stImage"] img { max-height: 62vh; object-fit: contain; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
#  Constants
# ============================================================
CATEGORIES = {
    "Games": ["Fortnite", "Minecraft", "FIFA", "Call of Duty", "Elden Ring"],
    "Food": ["Pizza", "Burger", "Salad", "Pasta"],
    "Animals": ["Dog", "Cat", "Lion", "Elephant", "Penguin"],
    "Vehicles": ["Car", "Bike", "Plane", "Ship", "Train"],
}
MAIN_MENU = ["Buy", "Remove from Stack", "Accept Order", "Cancel Order"]
REORDER_MENU = ["New Order", "Continue Shopping"]

# Gesture -> icon used both in the legend and on the buttons
GESTURE_ICONS = {
    "Next": "👍",
    "Previous": "👎",
    "Accept": "☝️",
    "Back": "🖐️",
    "Idle": "✊",
}

DETECTION_INTERVAL = 1.0  # seconds between gesture predictions


def get_categories():
    return list(CATEGORIES.keys())


def get_items(category):
    return CATEGORIES.get(category, [])


# ============================================================
#  Session state initialisation
# ============================================================
def init_state():
    defaults = {
        "stage": "main_menu",
        "menu_idx": 0,
        "cat_idx": 0,
        "item_idx": 0,
        "selected_category": None,
        "order_items": [],
        "remove_idx": 0,
        "camera_on": False,
        "last_status": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ============================================================
#  Gesture / action handler  (shared by mouse + camera)
# ============================================================
def process_gesture(gesture):
    """Apply an action. `gesture` is one of Next/Previous/Accept/Back."""
    s = st.session_state
    stage = s.stage

    if stage == "main_menu":
        if gesture == "Next":
            s.menu_idx = (s.menu_idx + 1) % len(MAIN_MENU)
        elif gesture == "Previous":
            s.menu_idx = (s.menu_idx - 1) % len(MAIN_MENU)
        elif gesture == "Accept":
            choice = MAIN_MENU[s.menu_idx]
            if choice == "Buy":
                s.stage = "selecting_category"
                s.cat_idx = 0
            elif choice == "Remove from Stack":
                s.stage = "remove_from_stack"
                s.remove_idx = 0
            elif choice == "Accept Order":
                s.stage = "order_summary"
            elif choice == "Cancel Order":
                s.order_items = []
                s.menu_idx = 0
                s.last_status = "Order cancelled."

    elif stage == "selecting_category":
        cats = get_categories()
        if gesture == "Next":
            s.cat_idx = (s.cat_idx + 1) % len(cats)
        elif gesture == "Previous":
            s.cat_idx = (s.cat_idx - 1) % len(cats)
        elif gesture == "Accept":
            s.selected_category = cats[s.cat_idx]
            s.stage = "selecting_item"
            s.item_idx = 0
        elif gesture == "Back":
            s.stage = "main_menu"

    elif stage == "selecting_item":
        items = get_items(s.selected_category)
        if gesture == "Next":
            s.item_idx = (s.item_idx + 1) % len(items)
        elif gesture == "Previous":
            s.item_idx = (s.item_idx - 1) % len(items)
        elif gesture == "Accept":
            s.order_items.append(f"{items[s.item_idx]} ({s.selected_category})")
            s.last_status = f"Added {items[s.item_idx]} to cart."
        elif gesture == "Back":
            s.stage = "selecting_category"

    elif stage == "remove_from_stack":
        if not s.order_items:
            if gesture == "Back":
                s.stage = "main_menu"
            return
        if gesture == "Next":
            s.remove_idx = (s.remove_idx + 1) % len(s.order_items)
        elif gesture == "Previous":
            s.remove_idx = (s.remove_idx - 1) % len(s.order_items)
        elif gesture == "Accept":
            removed = s.order_items.pop(s.remove_idx)
            s.last_status = f"Removed {removed}."
            if not s.order_items:
                s.stage = "main_menu"
            else:
                s.remove_idx = 0
        elif gesture == "Back":
            s.stage = "main_menu"

    elif stage == "order_summary":
        if gesture == "Accept":
            s.stage = "reorder_menu"
            s.menu_idx = 0
        elif gesture == "Back":
            s.stage = "main_menu"

    elif stage == "reorder_menu":
        if gesture == "Next":
            s.menu_idx = (s.menu_idx + 1) % len(REORDER_MENU)
        elif gesture == "Previous":
            s.menu_idx = (s.menu_idx - 1) % len(REORDER_MENU)
        elif gesture == "Accept":
            if REORDER_MENU[s.menu_idx] == "New Order":
                s.order_items = []
                s.stage = "main_menu"
                s.menu_idx = 0
                s.cat_idx = 0
                s.item_idx = 0
                s.selected_category = None
                s.remove_idx = 0
                s.last_status = "Started a new order."
            else:  # Continue Shopping
                s.stage = "selecting_category"
                s.cat_idx = 0
                s.item_idx = 0


# ============================================================
#  Stage description helpers
# ============================================================
def current_view():
    """Return (title, main_text, hint) for the active stage."""
    s = st.session_state
    stage = s.stage

    if stage == "main_menu":
        return "MAIN MENU", MAIN_MENU[s.menu_idx], f"{s.menu_idx + 1}/{len(MAIN_MENU)}"
    if stage == "selecting_category":
        cats = get_categories()
        return "SELECT CATEGORY", cats[s.cat_idx], f"{s.cat_idx + 1}/{len(cats)}"
    if stage == "selecting_item":
        items = get_items(s.selected_category)
        return (
            f"SELECT ITEM · {s.selected_category}",
            items[s.item_idx],
            f"{s.item_idx + 1}/{len(items)}",
        )
    if stage == "remove_from_stack":
        if s.order_items:
            return "REMOVE ITEM", s.order_items[s.remove_idx], f"{s.remove_idx + 1}/{len(s.order_items)}"
        return "REMOVE ITEM", "No items in cart", ""
    if stage == "order_summary":
        return "ORDER SUMMARY", f"{len(s.order_items)} item(s)", ""
    if stage == "reorder_menu":
        return "ORDER COMPLETE!", REORDER_MENU[s.menu_idx], f"{s.menu_idx + 1}/{len(REORDER_MENU)}"
    return "", "", ""


def available_actions():
    """Which gestures make sense for the current stage (controls buttons)."""
    stage = st.session_state.stage
    if stage in ("main_menu",):
        return ["Previous", "Next", "Accept"]
    if stage in ("selecting_category", "selecting_item", "reorder_menu"):
        return ["Previous", "Next", "Accept", "Back"]
    if stage == "remove_from_stack":
        if st.session_state.order_items:
            return ["Previous", "Next", "Accept", "Back"]
        return ["Back"]
    if stage == "order_summary":
        return ["Accept", "Back"]
    return []


# Friendlier verbs for the buttons depending on context
def action_verb(action):
    stage = st.session_state.stage
    verbs = {"Next": "Next", "Previous": "Previous", "Accept": "Select", "Back": "Back"}
    if action == "Accept":
        if stage in ("selecting_item",):
            verbs["Accept"] = "Add to cart"
        elif stage == "remove_from_stack":
            verbs["Accept"] = "Remove"
        elif stage == "order_summary":
            verbs["Accept"] = "Confirm"
    return verbs[action]


def action_label(action):
    return f"{GESTURE_ICONS[action]} {action_verb(action)}"


# ============================================================
#  Background camera worker (thread-safe gesture detection)
# ============================================================
@st.cache_resource
def get_camera_worker():
    """A single shared worker that owns the webcam and detects gestures.

    Lives across reruns thanks to cache_resource. The main thread reads
    `latest_frame` for display and pops `pending_gesture` to apply actions.
    """

    class CameraWorker:
        def __init__(self):
            self.lock = threading.Lock()
            self.latest_frame = None
            self.pending_gesture = None
            self.last_detected = None
            self.last_prediction = "—"   # raw label (incl. Idle / No hand) for display
            self.running = False
            self._thread = None
            self._last_pred_time = 0.0

        def start(self):
            # Guard against spawning a second reader (the root cause of the
            # MSMF "can't grab frame" spam = two threads fighting the webcam).
            with self.lock:
                if self.running and self._thread is not None and self._thread.is_alive():
                    return
                self.running = True
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()

        def stop(self):
            self.running = False

        def _open(self):
            # DirectShow is much quieter than MSMF on Windows.
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(0)  # fall back to default backend
            return cap

        def _loop(self):
            cap = self._open()
            fail_count = 0
            try:
                while self.running:
                    if not cap.isOpened():
                        time.sleep(0.1)
                        cap.release()
                        cap = self._open()
                        continue

                    ret, frame = cap.read()
                    if not ret:
                        # Recover from transient grab failures instead of
                        # hammering the device (which produces the warning flood).
                        fail_count += 1
                        if fail_count > 30:
                            cap.release()
                            cap = self._open()
                            fail_count = 0
                        time.sleep(0.05)
                        continue
                    fail_count = 0

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with self.lock:
                        self.latest_frame = rgb

                    now = time.time()
                    if now - self._last_pred_time >= DETECTION_INTERVAL:
                        self._last_pred_time = now
                        try:
                            landmarks = get_landmarks(rgb)
                            if landmarks:
                                flat = flatten_landmarks(landmarks[0])
                                gesture = predict_svc(flat)
                                with self.lock:
                                    self.last_prediction = gesture
                                if gesture != "Idle" and gesture != self.last_detected:
                                    with self.lock:
                                        self.pending_gesture = gesture
                                self.last_detected = gesture
                            else:
                                with self.lock:
                                    self.last_prediction = "No hand"
                                self.last_detected = None
                        except Exception:
                            pass
                    time.sleep(0.03)
            finally:
                cap.release()
                with self.lock:
                    self.latest_frame = None

        def read(self):
            with self.lock:
                return self.latest_frame

        def pop_gesture(self):
            with self.lock:
                g = self.pending_gesture
                self.pending_gesture = None
                return g

        def get_prediction(self):
            with self.lock:
                return self.last_prediction

    return CameraWorker()


worker = get_camera_worker()

# ============================================================
#  Sidebar — controls & legend
# ============================================================
with st.sidebar:
    st.header("🎮 Controls")
    st.session_state.camera_on = st.toggle("Enable camera & gestures", value=st.session_state.camera_on)
    if st.session_state.camera_on:
        worker.start()
    else:
        worker.stop()

    st.divider()
    st.subheader("Gesture Legend")
    legend = " ".join(
        f"<span class='gesture-badge'>{icon} {name}</span>"
        for name, icon in GESTURE_ICONS.items()
    )
    st.markdown(legend, unsafe_allow_html=True)
    st.caption(
        "The menu sits inside a D-pad: 👍 **Next** is on its right, "
        "👎 **Previous** on its left, 🖐️ **Back** on top, ☝️ **Select** below. "
        "Click a button or show the matching hand gesture."
    )

# ============================================================
#  Build the info panels as HTML strings (so they can be
#  re-rendered in place without redrawing the whole page).
#  The menu card sits in the centre of the D-pad cross; the
#  status + cart render below the whole console.
# ============================================================
def build_card_html():
    """The menu card that lives in the centre of the D-pad cross."""
    title, main_text, hint = current_view()
    return f"""
    <div class="stage-card">
        <div class="stage-title">{title}</div>
        <div class="stage-main">{main_text}</div>
        <div class="stage-hint">{hint}</div>
    </div>
    """


def build_cart_html():
    """Status banner + cart list, shown beneath the console."""
    s = st.session_state

    if s.order_items:
        cart_rows = "".join(
            f"<div style='padding:2px 0'>{i}. {item}</div>"
            for i, item in enumerate(s.order_items, 1)
        )
    else:
        cart_rows = "<div style='color:#888'>Empty</div>"

    status_html = (
        f"<div style='background:#143d14;border:1px solid #2e7d2e;border-radius:8px;"
        f"padding:6px 10px;margin:8px 0;font-size:14px'>✅ {s.last_status}</div>"
        if s.last_status
        else ""
    )

    return f"""
    {status_html}
    <div style='font-weight:700;font-size:16px;margin:6px 0'>🛒 Cart ({len(s.order_items)})</div>
    {cart_rows}
    """


# ============================================================
#  Main layout  (drawn once per rerun; the camera + info
#  panel are then updated in place inside the loop below)
# ============================================================
st.title("🎮 Gesture Order System")

left, right = st.columns([3, 2], gap="large")

with left:
    cam_box = st.empty()
    pred_box = st.empty()

with right:
    # The whole D-pad console (Back on top, Prev | menu | Next in the middle,
    # Select on the bottom) lives in one placeholder so it can be redrawn in
    # place when a gesture changes the stage — without an st.rerun() that would
    # freeze the video feed. The cart panel sits below it in its own placeholder.
    console_box = st.empty()
    cart_box = st.empty()


def render_prediction(label):
    """Show the live predicted gesture under the camera."""
    icon = GESTURE_ICONS.get(label, "🤚" if label == "No hand" else "")
    color = "#888" if label in ("No hand", "Idle", "—") else "#9aff9a"
    pred_box.markdown(
        f"<div style='text-align:center;font-size:18px;margin:4px 0'>"
        f"Predicted gesture: <b style='color:{color}'>{icon} {label}</b></div>",
        unsafe_allow_html=True,
    )


def _ghost(tall=False):
    """An empty, dimmed slot for a direction that isn't usable on this stage,
    so the cross keeps its shape and the user sees which gestures do nothing."""
    cls = "dpad-ghost tall" if tall else "dpad-ghost"
    st.markdown(f"<div class='{cls}'>·</div>", unsafe_allow_html=True)


def render_console():
    """Draw the D-pad console: the menu card wrapped by directional buttons.

        ┌──────────────────────────┐
        │        🖐️  Back          │   (top)
        │  👎  ┌────────────┐  👍   │
        │ Prev │  MENU CARD │ Next  │   (middle)
        │      └────────────┘       │
        │        ☝️  Select         │   (bottom)
        └──────────────────────────┘

    Each gesture's button is placed on the side that matches the gesture's
    direction so the user instantly correlates gesture → button → menu effect.
    Mouse clicks rerun normally (instant); on a *gesture* we redraw this in
    place so the video never freezes.
    """
    # Bump a counter so re-rendered buttons never collide on `key`
    # (Streamlit keeps old keys registered within the same script run).
    render_console._n = getattr(render_console, "_n", 0) + 1
    n = render_console._n
    actions = available_actions()

    def slot(action, tall=False):
        if action in actions:
            if st.button(action_label(action), key=f"btn_{action}_{n}"):
                process_gesture(action)
                st.rerun()
        else:
            _ghost(tall=tall)

    with console_box.container():
        # --- Top: Back ---
        _, top_c, _ = st.columns([1, 4, 1])
        with top_c:
            slot("Back")

        # --- Middle: Previous | menu card | Next ---
        c_prev, c_card, c_next = st.columns([1, 4, 1], gap="small")
        with c_prev:
            # Keyed container -> `st-key-dpad-prev` class -> tall-button CSS.
            with st.container(key=f"dpad-prev-{n}"):
                slot("Previous", tall=True)
        with c_card:
            st.markdown(build_card_html(), unsafe_allow_html=True)
        with c_next:
            with st.container(key=f"dpad-next-{n}"):
                slot("Next", tall=True)

        # --- Bottom: Select / Accept ---
        _, bot_c, _ = st.columns([1, 4, 1])
        with bot_c:
            slot("Accept")


def console_state():
    """Everything that affects which buttons exist + the menu card content.
    Used to decide when the in-place console needs a redraw."""
    return (st.session_state.stage, build_card_html())


# Initial render
render_console()
cart_box.markdown(build_cart_html(), unsafe_allow_html=True)

# ============================================================
#  Live loop — updates camera AND the D-pad console IN PLACE.
#  No st.rerun() on gesture, so the video feed never freezes.
# ============================================================
if not st.session_state.camera_on:
    cam_box.info("📷 Camera is off. Enable it in the sidebar to use gestures — or just use the buttons around the menu.")
else:
    last_console = console_state()
    last_cart = build_cart_html()
    last_pred = None
    while st.session_state.camera_on:
        frame = worker.read()
        if frame is not None:
            cam_box.image(frame, channels="RGB", use_container_width=True)
        else:
            cam_box.info("📷 Starting camera…")

        # Show the live predicted gesture (update only when it changes)
        pred = worker.get_prediction()
        if pred != last_pred:
            render_prediction(pred)
            last_pred = pred

        g = worker.pop_gesture()
        if g and g in ("Next", "Previous", "Accept", "Back"):
            process_gesture(g)

        # Redraw the whole console in place when either the stage (which
        # buttons exist) or the card content changes. One placeholder keeps
        # the buttons glued to the menu they act on.
        new_console = console_state()
        if new_console != last_console:
            render_console()
            last_console = new_console

        # Refresh the cart/status panel in place only if it changed.
        new_cart = build_cart_html()
        if new_cart != last_cart:
            cart_box.markdown(new_cart, unsafe_allow_html=True)
            last_cart = new_cart

        time.sleep(0.03)
