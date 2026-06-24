import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os

# from SvmModel import predict_svc

# model_path = 'gesture_recognizer.task'

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gesture_recognizer.task')

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)


mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles


def get_landmarks(image):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    recognition_result = recognizer.recognize(mp_image)

    hand_landmarks = getattr(recognition_result, 'hand_landmarks', None)

    if hand_landmarks:
        return hand_landmarks
    else:
        return None
    
def flatten_landmarks(landmarks):
    flattened = []
    for landmark in landmarks:
        flattened.append(landmark.x)
        flattened.append(landmark.y)
        flattened.append(landmark.z)
    return flattened


# cap = cv2.VideoCapture(0)



# try:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert the frame to RGB format
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Convert the RGB frame to a numpy array
#         numpy_frame_from_opencv = np.array(rgb_frame)


#         # Copy frame for annotations
#         annotated_frame = frame.copy()

#         # Draw skeleton landmarks when available
#         hand_landmarks = get_landmarks(numpy_frame_from_opencv)
#         if hand_landmarks is not None:
#             for landmarks in hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     annotated_frame,
#                     landmarks,
#                     mp_hands.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style())
                
#             inputLand = flatten_landmarks(hand_landmarks[0])

#             # Predict gesture using SVC model
#             gesture_prediction = predict_svc(inputLand)

#             print(f"Predicted gesture: {gesture_prediction}")


#         # Show annotated camera feed
#         cv2.imshow('Camera', annotated_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break


# except Exception as e:
#     print(f"An error occurred: {e}")
# except KeyboardInterrupt:
#     print("Video capture interrupted by user.")
# finally:
#     cap.release()
#     cv2.destroyAllWindows()