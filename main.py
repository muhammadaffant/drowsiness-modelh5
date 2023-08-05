import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
from flask import Flask, render_template, Response, request, jsonify
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

mixer.init()
sound = mixer.Sound('alert.mp3')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
model = None

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0

def load_my_model():
    global model
    model = load_model(os.path.join("models", "model.h5"))

def detect_eyes(frame):
    global score
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, minNeighbors=3, scaleFactor=1.1, minSize=(25, 25))
    eyes = eye_cascade.detectMultiScale(gray, minNeighbors=1, scaleFactor=1.1)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    for (x, y, w, h) in eyes:
        eye = frame[y:y + h, x:x + w]
        eye = cv2.resize(eye, (80, 80))
        eye = eye / 255
        eye = eye.reshape(80, 80, 3)
        eye = np.expand_dims(eye, axis=0)
        prediction = model.predict(eye)
        print(prediction)
        # Condition for Close
        if prediction[0][0] > 0.30:
            cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            score = score + 1
            if score > 10:
                try:
                    sound.play()
                except:
                    pass

        # Condition for Open
        elif prediction[0][1] > 0.70:
            score = score - 1
            if score < 0:
                score = 0
            cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    return frame

def generate_frames():
    load_my_model()  # Memuat model sebelum memulai deteksi mata
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_eyes(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


class EyeDetection(Resource):
    def post(self):
        frame = request.files['frame']
        frame = cv2.imdecode(np.fromstring(frame.read(), np.uint8), cv2.IMREAD_COLOR)
        processed_frame = detect_eyes(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        processed_frame = buffer.tobytes()
        return jsonify({'frame': processed_frame.tolist()})

api.add_resource(EyeDetection, '/detect_eyes')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    load_my_model()
    return jsonify({'message': 'Model loaded successfully'})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

cap.release()
cv2.destroyAllWindows()
