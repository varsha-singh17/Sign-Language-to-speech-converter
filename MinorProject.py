import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk 
import speech_recognition as sr
import os
import mediapipe as mp
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
import pyttsx3

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something:")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        result_label.config(text=f"Speech to Text: {text}",)
    except sr.UnknownValueError:
        result_label.config(text="Could not understand audio")
    except sr.RequestError as e:
        result_label.config(text=f"Error connecting to Google API: {e}")

def sign_language_to_speech():
    # Implement sign language to speech functionality here
    # This may involve using a sign language recognition model or API
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results
    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
    def draw_styled_landmarks(image, results):
    # Draw face connections
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                ) 
        # Draw pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                ) 
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                ) 
    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])
    # Path for exported data, numpy arrays
    DATA_PATH = os.path.join('MP_Data') 

    # Actions that we try to detect
    actions = np.array(['hello', 'thanks','I like it','A','B'])

    # Thirty videos worth of data
    no_sequences = 30

    # Videos are going to be 30 frames in length
    sequence_length = 30
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import TensorBoard

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.load_weights('Untitled Folder/action.h5')

    colors = [(245,117,16), (117,245,16), (16,117,245), (255,0,0), (0,255,0)]
    def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, (action, prob) in enumerate(zip(actions, res)):
            cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
            cv2.putText(output_frame, f"{action}: {prob:.2f}", (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return output_frame

    sequence = []
    sentence = []
    threshold = 0.8

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        last_spoken_time = time.time() - 5  # Initialize the last spoken time to ensure the first sign is spoken
        delay_duration = 5  # Delay duration in seconds

        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
                # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
    #         sequence.insert(0,keypoints)
    #         sequence = sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                
                
            #3. Viz logic
                if res[np.argmax(res)] > threshold: 
                    current_time = time.time()
                    # Check if 5 seconds have passed since the last spoken sign
                    if current_time - last_spoken_time >= delay_duration:
                        speak_text = actions[np.argmax(res)]
                        result_label.config(text=speak_text)
                        # Use pyttsx3 to speak the detected sign
                        engine = pyttsx3.init()
                        engine.say(speak_text)
                        engine.runAndWait()
                        last_spoken_time = current_time

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# Create main window
root = tk.Tk()
root.title("Speech and Sign Language Converter")

# Create and configure a frame
frame = ttk.Frame(root, padding="100")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
frame.columnconfigure(0, weight=1)
frame.rowconfigure(0, weight=1)
frame_style = ttk.Style()
frame_style.configure('TFrame', background='#ADD8E6')  # Light Blue color
# Load the logo image
# logo_image = Image.open("opencv/Logo-jiit.png")  # Replace with the path to your logo image
# logo_image = logo_image.resize((200, 100), Image.LANCZOS) 
# logo_photo = ImageTk.PhotoImage(logo_image)

# # Create a label for the logo
# logo_label = ttk.Label(frame, image=logo_photo, style='TLabel')
# logo_label.grid(row=0, column=0, pady=10, sticky=tk.W+tk.E+tk.N+tk.S)

heading_label = ttk.Label(frame, text="Sign Language Translator", style='Heading.TLabel')
heading_label.grid(row=0, column=0, pady=20)

# Create a style for the heading label
heading_style = ttk.Style()
heading_style.configure('Heading.TLabel', font=('Times New Roman', 25, 'bold'), background='#ADD8E6')  # Bold font and the same background color
# Create a style for the buttons
button_style = ttk.Style()
button_style.configure('TButton', font=('Times New Roman', 20), padding=10, foreground='blue', background='black',border=100)

# Create buttons with the specified style
speech_to_text_button = ttk.Button(frame, text="Speech to Text", command=speech_to_text, style='TButton')
speech_to_text_button.grid(row=2, column=0, pady=10, sticky=tk.W+tk.E+tk.N+tk.S)

sign_language_to_speech_button = ttk.Button(frame, text="Sign Language to Speech", command=sign_language_to_speech, style='TButton')
sign_language_to_speech_button.grid(row=3, column=0, pady=10, sticky=tk.W+tk.E+tk.N+tk.S)

# Create a label to display results
result_label = ttk.Label(frame, text="")
result_label.grid(row=4, column=0, pady=10)

# Start the Tkinter event loop
root.mainloop()