from flask import Flask, render_template, Response, request
from imutils.video import VideoStream
import imutils
import numpy as np
import cv2
from multiprocessing import Process
from multiprocessing import Queue
import time

app = Flask(__name__)

# run with raspberry pi usb camera or with built-in laptop webcam
pi = False

# detect with neural network or with haar cascades
network = True


def classify_frame_haar(input_queue, output_queue):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while True:        
        # check to see if there is a frame in our input queue
        if not input_queue.empty():
            frame = input_queue.get()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = face_cascade.detectMultiScale(gray, 1.3, 5)
            # write the detections to the output queue
            output_queue.put(detections)


def classify_frame_net(input_queue, output_queue):
    prototxt = "deploy.prototxt.txt"
    model = "res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    while True:        
        # check to see if there is a frame in our input queue
        if not input_queue.empty():
            frame = input_queue.get()
            # grab the frame from the input queue, resize it, and
            # construct a blob from it
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))

            # set the blob as input to our deep learning object
            # detector and obtain the detections
            net.setInput(blob)
            detections = net.forward()

            # write the detections to the output queue
            output_queue.put(detections)


@app.route('/')
def index():
    return render_template('index.html')


def process_frame(cam, input_queue, output_queue, detections):
    while True:
        if pi:
            frame = cam.read()
        else:
            _, frame = cam.read()
        # load the input image and construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        (h, w) = frame.shape[:2]

        if input_queue.empty():
            input_queue.put(frame)

        if not output_queue.empty():
            detections = output_queue.get()

        if detections is not None:
            if network:
                for i in range(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with the prediction
                    frame_confidence = detections[0, 0, i, 2]

                    # filter out weak detections by ensuring the `confidence` is
                    # greater than the minimum confidence
                    if frame_confidence > confidence:
                        # compute the (x, y)-coordinates of the bounding box for the object
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # draw the bounding box of the face
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            else:
                for (x, y, w, h) in detections:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        _, jpeg_frame = cv2.imencode('.jpg', frame)
        bytes_frame = jpeg_frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytes_frame + b'\r\n\r\n')


@app.route('/video_viewer', methods=['GET', 'POST'])
def video_viewer():
    global p
    if request.method == 'POST':
        json = request.get_json()
        mode = json['mode']
        if mode == "on":
            if p is None or not p.is_alive():
                # construct a child process *indepedent* from our main process of execution
                print("[INFO] starting process...")
                if network:
                    p = Process(target=classify_frame_net, args=(input_queue, output_queue,))
                else:
                    p = Process(target=classify_frame_haar, args=(input_queue, output_queue,))
                p.daemon = True
                p.start()
            return Response(process_frame(video_stream, input_queue, output_queue, detections),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            p.terminate()
            return '', 204
    else:
        return Response(process_frame(video_stream, input_queue, output_queue, detections),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    confidence = 0.5

    input_queue = Queue(maxsize=1)
    output_queue = Queue(maxsize=1)
    detections = None
    p = None

    print("[INFO] starting video stream...")
    video_stream = VideoStream(src=0).start() if pi else cv2.VideoCapture(0)
    time.sleep(2.0)
    app.run(host='0.0.0.0')
