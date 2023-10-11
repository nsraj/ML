from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import threading
import time

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('best.pt')

# Global variable to track weapon detection state
weapon_detection_active = False

# Thread for performing weapon detection
class WeaponDetectionThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.video_capture = cv2.VideoCapture('gunvid3.mp4')
        self.stopped = False

    def run(self):
        global weapon_detection_active

        while not self.stopped:
            if weapon_detection_active:
                # Read a frame from the video
                ret, frame = self.video_capture.read()

                if ret:
                    # Run YOLOv8 inference on the frame
                    results = model(frame)

                    # Visualize the results on the frame
                    annotated_frame = results[0].plot()

                    # Convert the annotated frame to JPEG format
                    ret, jpeg = cv2.imencode('.jpg', annotated_frame)

                    # Yield the JPEG frame as a byte array
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

            time.sleep(0.01)

        # Release the video capture object
        self.video_capture.release()

    def stop(self):
        self.stopped = True

# Create a global instance of the weapon detection thread
weapon_detection_thread = WeaponDetectionThread()
weapon_detection_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(weapon_detection_thread.run(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    global weapon_detection_active

    if not weapon_detection_active:
        weapon_detection_active = True

    return 'Weapon detection started.'

@app.route('/stop_detection')
def stop_detection():
    global weapon_detection_active

    if weapon_detection_active:
        weapon_detection_active = False

    return 'Weapon detection stopped.'

if __name__ == '__main__':
    app.run(debug=True)
