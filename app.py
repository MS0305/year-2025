from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp

app = Flask(__name__)

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Global variable to store finger count
finger_count_global = 0

# Function to process the video and calculate the finger count
def generate_frames():
    global finger_count_global  # Declare global variable
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)
            finger_count = 0

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = hand_landmarks.landmark

                    # Thumb logic
                    if landmarks[4].x < landmarks[3].x:
                        finger_count += 1

                    # Other fingers logic
                    for tip in [8, 12, 16, 20]:
                        if landmarks[tip].y < landmarks[tip - 2].y:
                            finger_count += 1

            # Update the global finger count
            finger_count_global = finger_count

            # Encode the frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_finger_count')
def get_finger_count():
    global finger_count_global
    return jsonify({"finger_count": finger_count_global})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
