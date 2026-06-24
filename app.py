import streamlit as st
import cv2
import time
import numpy as np

from Model.MediapipeModel import get_landmarks, flatten_landmarks
from Model.SvmModel import predict_svc

# ---------- Page config ----------
st.set_page_config(layout="centered", initial_sidebar_state="collapsed")
st.markdown("""<style>html{scroll-behavior:auto!important}body{overflow-y:auto!important}</style>""", unsafe_allow_html=True)
st.title("🎮 Gesture Order System")
# Gesture legend (static, shown below title)
st.markdown("""
---
**Gesture Legend**  
👍 Next &nbsp;|&nbsp; 👎 Previous &nbsp;|&nbsp; ☝️ Accept &nbsp;|&nbsp; 🖐️ Back &nbsp;|&nbsp; ✊ Idle
""")

# ---------- Constants ----------
CATEGORIES = {
    "Games": ["Fortnite", "Minecraft", "FIFA", "Call of Duty", "Elden Ring"],
    "Food": ["Pizza", "Burger", "Salad","Pasta"],
    "Animals": ["Dog", "Cat", "Lion", "Elephant", "Penguin"],
    "Vehicles": ["Car", "Bike", "Plane", "Ship", "Train"]
}
MAIN_MENU = ["Buy", "Remove from Stack", "Accept Order", "Cancel Order"]
REORDER_MENU = ["New Order", "Continue Shopping"]

# ---------- Helper functions ----------
def get_categories():
    return list(CATEGORIES.keys())

def get_items(category):
    return CATEGORIES.get(category, [])

# ---------- State variables (plain Python) ----------
stage = 'main_menu'
menu_idx = 0
cat_idx = 0
item_idx = 0
selected_category = None
order_items = []          # plain list – no session state!
remove_idx = 0
last_gesture = None

# ---------- Gesture handler ----------
def process_gesture(gesture):
    global stage, menu_idx, cat_idx, item_idx, selected_category, order_items, remove_idx, last_gesture

    # Main menu
    if stage == 'main_menu':
        if gesture == "Next":
            menu_idx = (menu_idx + 1) % len(MAIN_MENU)
        elif gesture == "Previous":
            menu_idx = (menu_idx - 1) % len(MAIN_MENU)
        elif gesture == "Accept":
            choice = MAIN_MENU[menu_idx]
            if choice == "Buy":
                stage = 'selecting_category'
                cat_idx = 0
            elif choice == "Remove from Stack":
                stage = 'remove_from_stack'
                remove_idx = 0
            elif choice == "Accept Order":
                stage = 'order_summary'
            elif choice == "Cancel Order":
                order_items.clear()
                menu_idx = 0
        last_gesture = None

    # Selecting category
    elif stage == 'selecting_category':
        cats = get_categories()
        if gesture == "Next":
            cat_idx = (cat_idx + 1) % len(cats)
        elif gesture == "Previous":
            cat_idx = (cat_idx - 1) % len(cats)
        elif gesture == "Accept":
            selected_category = cats[cat_idx]
            stage = 'selecting_item'
            item_idx = 0
        elif gesture == "Back":
            stage = 'main_menu'
        last_gesture = None

    # Selecting item
    elif stage == 'selecting_item':
        items = get_items(selected_category)
        if gesture == "Next":
            item_idx = (item_idx + 1) % len(items)
        elif gesture == "Previous":
            item_idx = (item_idx - 1) % len(items)
        elif gesture == "Accept":
            order_items.append(f"{items[item_idx]} ({selected_category})")
        elif gesture == "Back":
            stage = 'selecting_category'
        last_gesture = None

    # Removing item
    elif stage == 'remove_from_stack':
        if not order_items:
            if gesture == "Back":
                stage = 'main_menu'
                last_gesture = None
            return

        if gesture == "Next":
            remove_idx = (remove_idx + 1) % len(order_items)
        elif gesture == "Previous":
            remove_idx = (remove_idx - 1) % len(order_items)
        elif gesture == "Accept":
            order_items.pop(remove_idx)
            if not order_items:
                stage = 'main_menu'
            else:
                remove_idx = 0
        elif gesture == "Back":
            stage = 'main_menu'
        last_gesture = None

    # Order summary
    elif stage == 'order_summary':
        if gesture == "Accept":
            stage = 'reorder_menu'
            menu_idx = 0
        elif gesture == "Back":
            stage = 'main_menu'
        last_gesture = None

    # Reorder menu
    elif stage == 'reorder_menu':
        if gesture == "Next":
            menu_idx = (menu_idx + 1) % len(REORDER_MENU)
        elif gesture == "Previous":
            menu_idx = (menu_idx - 1) % len(REORDER_MENU)
        elif gesture == "Accept":
            if REORDER_MENU[menu_idx] == "New Order":
                order_items.clear()
                stage = 'main_menu'
                menu_idx = 0
                cat_idx = 0
                item_idx = 0
                selected_category = None
                remove_idx = 0
            else:  # Continue Shopping
                stage = 'selecting_category'
                cat_idx = 0
                item_idx = 0
        last_gesture = None

# ---------- UI Layout ----------
col1, col2 = st.columns([3, 1])
with col1:
    frame_placeholder = st.empty()
with col2:
    info_placeholder = st.empty()   # single placeholder for the whole info panel

# ---------- Camera ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Cannot access camera!")
    st.stop()

last_detection = time.time()
DETECTION_INTERVAL = 2.0

# Cached info string to avoid blinking
last_info_html = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---------- Build info as a single markdown string ----------
    stage_text = main_text = hint_text = ""

    if stage == 'main_menu':
        stage_text = "MAIN MENU"
        main_text = MAIN_MENU[menu_idx]
        hint_text = f"({menu_idx+1}/{len(MAIN_MENU)})"
    elif stage == 'selecting_category':
        cats = get_categories()
        stage_text = "SELECT CATEGORY"
        main_text = cats[cat_idx]
        hint_text = f"({cat_idx+1}/{len(cats)})"
    elif stage == 'selecting_item':
        items = get_items(selected_category)
        stage_text = f"SELECT ITEM ({selected_category})"
        main_text = items[item_idx]
        hint_text = f"({item_idx+1}/{len(items)})"
    elif stage == 'remove_from_stack':
        stage_text = "REMOVE ITEM"
        if order_items:
            main_text = order_items[remove_idx]
            hint_text = f"({remove_idx+1}/{len(order_items)})"
        else:
            main_text = "NO ITEMS"
            hint_text = ""
    elif stage == 'order_summary':
        stage_text = "ORDER SUMMARY"
        main_text = f"{len(order_items)} item(s)"
        hint_text = ""
    elif stage == 'reorder_menu':
        stage_text = "ORDER COMPLETE!"
        main_text = REORDER_MENU[menu_idx]
        hint_text = f"({menu_idx+1}/{len(REORDER_MENU)})"

    # Build the cart list
    cart_lines = []
    if order_items:
        for i, item in enumerate(order_items, 1):
            cart_lines.append(f"{i}. {item}")
    else:
        cart_lines.append("Empty")

    cart_html = "**Cart ({})**\n\n".format(len(order_items)) + "\n".join(cart_lines)

#     legend_html = """
# ---
# **Gesture Legend**  
# 👍 Next &nbsp;|&nbsp; 👎 Previous &nbsp;|&nbsp; ☝️ Accept &nbsp;|&nbsp; 🖐️ Back &nbsp;|&nbsp; ✊ Idle
# """
    # Full info HTML
    info_html = f"""
### {stage_text}
**{main_text}**
{f'*{hint_text}*' if hint_text else ''}

---

{cart_html}


    """

    # Update the placeholder **only if the content changed**
    if info_html != last_info_html:
        info_placeholder.markdown(info_html)
        last_info_html = info_html

    # Update camera feed (always)
    frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

    # ---------- Gesture detection ----------
    now = time.time()
    if now - last_detection >= DETECTION_INTERVAL:
        try:
            landmarks = get_landmarks(rgb)
            if landmarks:
                flat = flatten_landmarks(landmarks[0])
                gesture = predict_svc(flat)
                if gesture != last_gesture:
                    last_gesture = gesture
                    process_gesture(gesture)
        except Exception:
            pass
        last_detection = now

    time.sleep(0.03)

cap.release()