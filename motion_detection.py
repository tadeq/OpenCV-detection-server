from flask import Flask, render_template, Response, request
from imutils.video import VideoStream
import imutils
import cv2
from multiprocessing import Process
from multiprocessing import Queue
import time

app = Flask(__name__)

# run with raspberry pi usb camera or with built-in laptop webcam
pi = False


@app.route('/')
def index():
    return render_template('welcome_page.html')


@app.route('/configure', methods=['GET', 'POST'])
def configure():
    return render_template('configure.html')


@app.route('/video_viewer_conf', methods=['GET', 'POST'])
def video_viewer_conf():
    global rectangle
    if request.method == 'POST':
        json = request.get_json()
        pixel_from_x = json['pixelFromX']
        pixel_from_y = json['pixelFromY']
        width = json['width']
        height = json['height']
        [pixel_from_x, pixel_from_y] = [val if val != '' else '0' for val in [pixel_from_x, pixel_from_y]]
        [width, height] = [val if val != '' else '5000' for val in [width, height]]
        [pixel_from_x, pixel_from_y, width, height] = [int(val) for val in [pixel_from_x, pixel_from_y, width, height]]
        [pixel_from_x, pixel_from_y] = [val if val < 5000 else 0 for val in [pixel_from_x, pixel_from_y]]
        [width, height] = [val if val < 5000 else 5000 for val in [width, height]]
        if not rectangle.empty():
            rectangle.get()
        rectangle.put([pixel_from_x, pixel_from_y, width, height])
        return Response()
    else:
        return Response(draw_area(video_stream), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/surveillance')
def surveillance():
    return render_template('index.html')


@app.route('/video_viewer', methods=['GET', 'POST'])
def video_viewer():
    global p
    if request.method == 'POST':
        json = request.get_json()
        mode = json['mode']
        if mode == "on":
            if p is None or not p.is_alive():
                print("[INFO] starting process...")
                p = Process(target=classify_frame, args=())
                p.daemon = True
                p.start()
            return Response(process_frame(video_stream),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            p.terminate()
            return '', 204
    else:
        return Response(process_frame(video_stream),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


def draw_area(cam):
    while True:
        frame = None
        if pi:
            frame = cam.read()
        else:
            _, frame = cam.read()

        if not rectangle.empty():
            det_area = rectangle.get()  # det_area-current value of the rectangle
            rectangle.put(det_area)
            # det_area[0]==pixel_from_x; det_area[1]==pixel_from_y; det_area[2]==width; det_area[3]==height
            cv2.rectangle(frame, (det_area[0], det_area[1]), (det_area[0] + det_area[2], det_area[1] + det_area[3]),
                          (0, 255, 0), 2)

        _, jpeg_frame = cv2.imencode('.jpg', frame)
        bytes_frame = jpeg_frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytes_frame + b'\r\n\r\n')


def classify_frame():
    global detections, first_frame
    while True:
        # check to see if there is a frame in our input queue
        if not input_queue.empty():
            frame = input_queue.get()

            if not rectangle.empty():
                det_area = rectangle.get()  # det_area-current value of the rectangle
                rectangle.put(det_area)
                # det_area[0]==pixel_from_x; det_area[1]==pixel_from_y; det_area[2]==width; det_area[3]==height
                frame = frame[det_area[0]:det_area[0] + det_area[2], det_area[1]:det_area[1] + det_area[3]]
            # resize the frame, convert it to grayscale, and blur it
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # if the first frame is None, initialize it
            if first_frame is None:
                first_frame = gray
                continue
            if len(frame) != len(first_frame):
                continue
            # compute the absolute difference between the current frame and first frame
            frame_delta = cv2.absdiff(first_frame, gray)
            thresh = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)[1]

            # dilate the thresholded image to fill in holes, then find contours on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            detections = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detections = imutils.grab_contours(detections)
            output_queue.put(detections)


def process_frame(cam):
    global detections
    while True:
        if pi:
            frame = cam.read()
        else:
            _, frame = cam.read()

        if input_queue.empty():
            input_queue.put(frame)

        if not output_queue.empty():
            detections = output_queue.get()

        if not rectangle.empty():
            det_area = rectangle.get()  # det_area-current value of the rectangle
            rectangle.put(det_area)
            # det_area[0]==pixel_from_x; det_area[1]==pixel_from_y; det_area[2]==width; det_area[3]==height
            cv2.rectangle(frame, (det_area[0], det_area[1]), (det_area[0] + det_area[2], det_area[1] + det_area[3]),
                          (0, 0, 255), 2)

        if detections is not None:
            for d in detections:
                # if cv2.contourArea(d) > 360:
                # compute the bounding box for the contour, draw it on the frame
                (x, y, w, h) = cv2.boundingRect(d)
                if not rectangle.empty():
                    rect = rectangle.get()
                    rectangle.put(rect)
                    cv2.rectangle(frame, (x + rect[0], y + rect[1]), (x + w, y + h),
                                  (0, 255, 0), 2)

        _, jpeg_frame = cv2.imencode('.jpg', frame)
        bytes_frame = jpeg_frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytes_frame + b'\r\n\r\n')


if __name__ == '__main__':
    input_queue = Queue(maxsize=1)
    output_queue = Queue(maxsize=1)
    first_frame = None
    detections = None
    p = None
    rectangle = Queue(maxsize=1)
    rectangle.put([0, 0, 0, 0])

    print("[INFO] starting video stream...")
    video_stream = VideoStream(src=0).start() if pi else cv2.VideoCapture(0)
    time.sleep(2.0)
    app.run(host='0.0.0.0')
