import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tensorflow.lite as tflite

# Initialize Mediapipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define constants
ROWS_PER_FRAME = 543

# Initialize ORD2SIGN (from train.csv)
train = pd.read_csv('train.csv')

# Create numeric labels for each unique sign using category codes
train['sign_ord'] = train['sign'].astype('category').cat.codes

# Create the SIGN2ORD mapping (sign name to numeric label)
SIGN2ORD = train[['sign', 'sign_ord']].drop_duplicates().set_index('sign').squeeze().to_dict()

# Create the ORD2SIGN mapping (numeric label to sign name)
ORD2SIGN = train[['sign_ord', 'sign']].drop_duplicates().set_index('sign_ord').squeeze().to_dict()


# Function to load the data
def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


# Function to create a DataFrame from the frame landmarks (same as your `create_frame_landmark_df`)
def create_frame_landmark_df(results, frame, xyz):
    key_skel = xyz[['type', 'landmark_index']].drop_duplicates().reset_index(drop=True).copy()

    face = pd.DataFrame()
    pose = pd.DataFrame()
    left_hand = pd.DataFrame()
    right_hand = pd.DataFrame()

    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    face = face.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='face')
    pose = pose.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='pose')
    left_hand = left_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='left_hand')
    right_hand = right_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='right_hand')

    landmarks = pd.concat([face, pose, left_hand, right_hand]).reset_index(drop=True)
    landmarks = key_skel.merge(landmarks, on=['type', 'landmark_index'], how='left')
    landmarks = landmarks.assign(frame=frame)

    return landmarks


def capture_video_and_process_landmarks(output_parquet_file):
    pq_file = 'train_landmark_files/16069/10042041.parquet'
    xyz = pd.read_parquet(pq_file)

    cap = cv2.VideoCapture(0)
    all_landmarks = []
    frame = 0

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            landmarks = create_frame_landmark_df(results, frame, xyz)
            all_landmarks.append(landmarks)

            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image_bgr, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
            mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            cv2.imshow('Live Video Feed', cv2.flip(image_bgr, 1))

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop capture
                break

        cap.release()
        cv2.destroyAllWindows()

    landmarks_df = pd.concat(all_landmarks).reset_index(drop=True)
    landmarks_df.to_parquet(output_parquet_file)


# Function to perform inference using the TFLite model
def perform_inference(xyz_np):
    interpreter = tflite.Interpreter("./model.tflite")
    prediction_fn = interpreter.get_signature_runner("serving_default")
    prediction = prediction_fn(inputs=xyz_np)
    sign = prediction['outputs'].argmax()  # Get the predicted numeric sign (index)
    return sign


# Create the GUI class for capturing and performing predictions
class SignLanguageCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Sign Language Recognition")
        self.root.geometry("400x300")

        # Button to start/stop capture
        self.start_button = tk.Button(root, text="Start Capture", command=self.start_capture)
        self.start_button.pack(pady=30)

        # Label to show the prediction result
        self.result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 50))
        self.result_label.pack(pady=20)

        # State to manage the capture process
        self.is_capturing = False
        self.is_running = False

    def start_capture(self):
        if self.is_capturing:
            self.is_capturing = False
            self.start_button.config(text="Start Capture")
            messagebox.showinfo("Info", "Capture Stopped!")
            self.stop_capture()
        else:
            self.is_capturing = True
            self.start_button.config(text="Stop Capture")
            messagebox.showinfo("Info", "Capture Started!")
            self.capture_and_process_landmarks()

    def capture_and_process_landmarks(self):
        self.is_running = True
        output_parquet_file = 'output.parquet'
        capture_video_and_process_landmarks(output_parquet_file)
        self.perform_prediction(output_parquet_file)

    def perform_prediction(self, output_parquet_file):
        # Load the captured landmarks from the saved parquet file
        xyz_np = load_relevant_data_subset(output_parquet_file)

        # Perform inference using the TFLite model
        sign = perform_inference(xyz_np)

        # Map the numeric prediction to the sign word using ORD2SIGN
        predicted_sign = ORD2SIGN.get(sign, "Unknown")  # Get the word, or "Unknown" if not found

        # Display the prediction result in the GUI
        prediction_result = f"Prediction: {predicted_sign}"  # Display the sign word
        self.result_label.config(text=prediction_result)

    def stop_capture(self):
        # Reset the video capture state
        self.is_running = False
        cv2.destroyAllWindows()


# Create the main window and run the Tkinter application
root = tk.Tk()
app = SignLanguageCaptureApp(root)
root.mainloop()
