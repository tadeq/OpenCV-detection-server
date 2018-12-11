from flask import Flask, render_template, Response
# from imutils.video import VideoStream
# import imutils
import numpy as np
import cv2
from multiprocessing import Process
from multiprocessing import Queue
import time

app = Flask(__name__)


def classify_frame(net, input_queue, output_queue):
    while True:
        # check to see if there is a frame in our input queue
        if not input_queue.empty():
            # grab the frame from the input queue, resize it, and
            # construct a blob from it
            frame = input_queue.get()
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


def video_stream():
    prototxt = "deploy.prototxt.txt"
    model = "res10_300x300_ssd_iter_140000.caffemodel"
    confidence = 0.5

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # initialize the input queue (frames), output queue (detections),
    # and the list of actual detections returned by the child process
    input_queue = Queue(maxsize=1)
    output_queue = Queue(maxsize=1)
    detections = None

    # construct a child process *indepedent* from our main process of
    # execution
    print("[INFO] starting process...")
    p = Process(target=classify_frame, args=(net, input_queue, output_queue,))
    p.daemon = True
    p.start()

    print("[INFO] starting video stream...")
    # vs = VideoStream(src=0).start()
    # vs = VideoStream(usePiCamera=True).start()
    vc = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        # image = vs.read()
        _, image = vc.read()
        # load the input image and construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        (h, w) = image.shape[:2]

        # if the input queue *is* empty, give the current frame to
        # classify
        if input_queue.empty():
            input_queue.put(image)

        # if the output queue *is not* empty, grab the detections
        if not output_queue.empty():
            detections = output_queue.get()

        # check to see if our detectios are not None (and if so, we'll
        # draw the detections on the frame)
        if detections is not None:
            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                frame_confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if frame_confidence > confidence:
                    # compute the (x, y)-coordinates of the bounding box for the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # draw the bounding box of the face
                    cv2.rectangle(image, (startX, startY), (endX, endY),
                                  (0, 0, 255), 2)

        _, jpeg_frame = cv2.imencode('.jpg', image)
        bytes_frame = jpeg_frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytes_frame + b'\r\n\r\n')


@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
