import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import pyfirmata2
import time

board = pyfirmata2.Arduino('COM4')

# Setup PWM pins for LED intensity control
red_led = board.get_pin('d:11:p')
green_led = board.get_pin('d:10:p')
yellow_led = board.get_pin('d:9:p')
blue_led = board.get_pin('d:6:p')

def detect_emotion():
    # Initialize necessary variables
    cap = cv2.VideoCapture(0)  # or the appropriate source
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classifier = load_model('model.h5')  # Load your trained model
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    previous_label = None
    start_time = None
    stability_duration = 5  # seconds

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                max_index = prediction.argmax()
                label = emotion_labels[max_index]
                intensity = prediction[max_index] * 255  # Scale intensity to 0-255 for PWM

                # Set LED colors based on the detected emotion
                if label == 'Angry':
                    red_led.write(intensity / 255.0)
                    green_led.write(0)
                    yellow_led.write(0)
                    blue_led.write(0)
                elif label == 'Neutral':
                    red_led.write(0)
                    green_led.write(intensity / 255.0)
                    yellow_led.write(0)
                    blue_led.write(0)
                elif label == 'Happy':
                    red_led.write(0)
                    green_led.write(0)
                    yellow_led.write(intensity / 255.0)
                    blue_led.write(0)
                elif label == 'Sad':
                    red_led.write(0)
                    green_led.write(0)
                    yellow_led.write(0)
                    blue_led.write(intensity / 255.0)

                # Check if the emotion is stable
                if label == previous_label:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= stability_duration:
                        intensity = 255
                        # pass  # Emotion is stable, but no need to slow down
                else:
                    previous_label = label
                    start_time = None

            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Reset LEDs to off after each loop iteration (handled by setting intensity to 0 in each condition)
        # No need for time.sleep(d)

    cap.release()
    cv2.destroyAllWindows()

detect_emotion()
