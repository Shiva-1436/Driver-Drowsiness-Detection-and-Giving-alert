# import cv2
# import dlib
# import numpy as np
# from scipy.spatial import distance
# import winsound  # For alert sound (Windows)
# import time  # To track time-based alerts

# # Load face detector and landmark predictor
# detector = dlib.get_frontal_face_detector()
# # predictor = dlib.shape_predictor(r"C:\Users\shiva\Downloads\shape_predictor_68_face_landmarks.dat")
# predictor = dlib.shape_predictor(r"C:\Users\shiva\Downloads\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")





# # Define eye and mouth landmark indexes
# LEFT_EYE = [36, 37, 38, 39, 40, 41]
# RIGHT_EYE = [42, 43, 44, 45, 46, 47]
# MOUTH = [60, 61, 62, 63, 64, 65, 66, 67]

# # Thresholds
# EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold
# MAR_THRESHOLD = 0.5  # Mouth Aspect Ratio threshold
# CLOSED_FRAMES = 90  # ~3 seconds at ~30 FPS

# EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold (below this = closed eyes)
# MAR_THRESHOLD = 0.6  # Mouth Aspect Ratio threshold (above this = yawning)
# CLOSED_DURATION = 3  # Time in seconds before triggering alert
# last_closed_time = None  # Track time when eyes were first closed
# frame_counter = 0  # Counter for eye closure

# # Function to compute Eye Aspect Ratio (EAR)
# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)

# # Function to compute Mouth Aspect Ratio (MAR)
# def mouth_aspect_ratio(mouth):
#     A = distance.euclidean(mouth[1], mouth[5])  
#     B = distance.euclidean(mouth[2], mouth[4])  
#     C = distance.euclidean(mouth[0], mouth[3])  
#     return (A + B) / (2.0 * C)

# # Function to alert driver
# def alert_driver():
#     global last_alert_time
#     current_time = time.time()
#     if current_time - last_alert_time >= alert_interval:  # Alert only if 3 seconds have passed
#         print("ALERT! Driver Drowsy or Yawning!")  
#         winsound.Beep(1000, 1000)  
#         last_alert_time = current_time  # Update last alert time

# # Start video capture
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     for face in faces:
#         landmarks = predictor(gray, face)

#         # Get eye and mouth coordinates
#         left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE])
#         right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE])
#         mouth = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in MOUTH])

#         # Compute EAR and MAR
#         left_ear = eye_aspect_ratio(left_eye)
#         right_ear = eye_aspect_ratio(right_eye)
#         ear = (left_ear + right_ear) / 2.0
#         mar = mouth_aspect_ratio(mouth)

#         # Draw landmarks
#         for (x, y) in np.concatenate((left_eye, right_eye, mouth), axis=0):
#             cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

#         # Check for drowsiness (eyes closed for 3 seconds)
#         if ear < EAR_THRESHOLD:
#             frame_counter += 1
#             if frame_counter >= CLOSED_FRAMES:
#                 cv2.putText(frame, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#                 alert_driver()  # Alert once every 3 seconds
#         else:
#             frame_counter = 0  # Reset counter if eyes open

#         # Check for yawning (mouth open)
#         if mar > MAR_THRESHOLD:
#             cv2.putText(frame, "YAWN ALERT!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#             alert_driver()  # Alert once every 3 seconds

#     # Display the video frame
#     cv2.imshow("Driver Monitoring", frame)

#     # Exit when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import time  # Added for tracking closed eye duration
import winsound  # For alert sound (Windows)

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\shiva\Downloads\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")

# Define eye and mouth landmark indexes
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
MOUTH = [60, 61, 62, 63, 64, 65, 66, 67]

# Thresholds and frame counters
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold (below this = closed eyes)
MAR_THRESHOLD = 0.6  # Mouth Aspect Ratio threshold (above this = yawning)
CLOSED_DURATION = 3  # Time in seconds before triggering alert
ALERT_INTERVAL = 3  # Minimum time between alerts

last_closed_time = None  # Track time when eyes were first closed
last_alert_time = 0  # Track last alert time

# Function to compute Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])  # Vertical distance
    B = distance.euclidean(eye[2], eye[4])  # Vertical distance
    C = distance.euclidean(eye[0], eye[3])  # Horizontal distance
    return (A + B) / (2.0 * C)

# Function to compute Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[1], mouth[5])  # Vertical distance
    B = distance.euclidean(mouth[2], mouth[4])  # Vertical distance
    C = distance.euclidean(mouth[0], mouth[3])  # Horizontal distance
    return (A + B) / (2.0 * C)

# Function to alert driver
def alert_driver():
    global last_alert_time
    current_time = time.time()
    if current_time - last_alert_time >= ALERT_INTERVAL:  # Alert only if 3 seconds have passed
        print("🚨 ALERT! Driver Drowsy or Yawning!")
        winsound.Beep(1000, 1000)  # Beep sound (1000Hz for 1 sec)
        last_alert_time = current_time  # Update last alert time

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Get coordinates for both eyes and mouth
        left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE])
        right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE])
        mouth = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in MOUTH])

        # Compute EAR and MAR
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Draw landmarks
        for (x, y) in np.concatenate((left_eye, right_eye, mouth), axis=0):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Check for drowsiness (Closed Eyes for 3 seconds)
        if ear < EAR_THRESHOLD:
            if last_closed_time is None:  # Eyes just closed
                last_closed_time = time.time()
            elif time.time() - last_closed_time >= CLOSED_DURATION:  # Closed for 3+ seconds
                cv2.putText(frame, "🚨 DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                alert_driver()  # Alert once every 3 seconds
        else:
            last_closed_time = None  # Reset counter if eyes are open

        # Check for yawning (Open mouth)
        if mar > MAR_THRESHOLD:
            cv2.putText(frame, "😮 YAWN ALERT!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            alert_driver()  # Alert once every 3 seconds

    # Display the video frame
    cv2.imshow("Driver Monitoring", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


# import cv2
# import dlib
# import numpy as np
# from scipy.spatial import distance
# import time  # For time tracking
# import winsound  # For Windows beep sound

# # Load face detector and landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(r"C:\Users\shiva\Downloads\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")

# # Define eye and mouth landmark indexes
# LEFT_EYE = [36, 37, 38, 39, 40, 41]
# RIGHT_EYE = [42, 43, 44, 45, 46, 47]
# MOUTH = [60, 61, 62, 63, 64, 65, 66, 67]

# # Thresholds
# EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold (closed eyes)
# MAR_THRESHOLD = 0.6  # Mouth Aspect Ratio threshold (yawning)
# CLOSED_DURATION = 3  # Time (seconds) before triggering alert
# FRAME_RATE = 30  # Assuming 30 FPS

# closed_frame_count = 0  # Count of consecutive closed-eye frames
# required_closed_frames = CLOSED_DURATION * FRAME_RATE  # Convert time to frames

# alert_active = False  # Flag to indicate if alert is active

# # Function to compute Eye Aspect Ratio (EAR)
# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)

# # Function to compute Mouth Aspect Ratio (MAR)
# def mouth_aspect_ratio(mouth):
#     A = distance.euclidean(mouth[1], mouth[5])
#     B = distance.euclidean(mouth[2], mouth[4])
#     C = distance.euclidean(mouth[0], mouth[3])
#     return (A + B) / (2.0 * C)

# # Function to alert driver continuously
# def alert_driver():
#     winsound.Beep(1000, 500)  # Beep sound (1000Hz for 0.5 sec)

# # Start video capture
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     for face in faces:
#         landmarks = predictor(gray, face)

#         # Get coordinates for both eyes and mouth
#         left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE])
#         right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE])
#         mouth = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in MOUTH])

#         # Compute EAR and MAR
#         left_ear = eye_aspect_ratio(left_eye)
#         right_ear = eye_aspect_ratio(right_eye)
#         ear = (left_ear + right_ear) / 2.0
#         mar = mouth_aspect_ratio(mouth)

#         # Draw landmarks
#         for (x, y) in np.concatenate((left_eye, right_eye, mouth), axis=0):
#             cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

#         # Check for drowsiness (Eyes closed for 3+ seconds)
#         if ear < EAR_THRESHOLD:
#             closed_frame_count += 1  # Increment if eyes are closed
#             if closed_frame_count >= required_closed_frames:
#                 alert_active = True
#                 cv2.putText(frame, "🚨 DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#                 alert_driver()  # Keep beeping as long as eyes are closed

#         else:
#             closed_frame_count = 0  # Reset if eyes are open
#             alert_active = False  # Stop alert when driver opens eyes

#         # Check for yawning (Mouth open)
#         if mar > MAR_THRESHOLD:
#             cv2.putText(frame, "😮 YAWN ALERT!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#             alert_driver()

#     # Display the video frame
#     cv2.imshow("Driver Monitoring", frame)

#     # Exit when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
