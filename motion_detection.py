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
network = False

@app.route('/')
def index():
    return render_template('welcome_page.html')

@app.route('/configure', methods=['GET', 'POST'])
def configure():
    return render_template('configure.html')

@app.route('/video_viewer_conf', methods=['GET', 'POST'])
def video_viewer_conf():    
    if request.method == 'POST':
        global rectangle
        json = request.get_json()
        pixelFromX = int(json['pixelFromX'])
        pixelFromY = int(json['pixelFromY'])
        width = int(json['width'])
        height = int(json['height']) 
        if not rectangle.empty():
            rectangle.get()
        rectangle.put([pixelFromX, pixelFromY, width, height])
        return Response()
    else:
        return Response(draw_area(video_stream, rectangle),mimetype='multipart/x-mixed-replace; boundary=frame')

def draw_area(cam, rectangle):
    while True:   
        frame = None    
        try:
            if pi:
                frame = cam.read()
            else:
                _, frame = cam.read()                
        except:                
            #One frame wasn't read properly            
            pass
        
        if frame is not None:  
            
            if not rectangle.empty():                
                rectV = rectangle.get() #rectV-current value of the rectangle
                rectangle.put(rectV)   
                #rectV[0]==pixelFromX; rectV[1]==pixelFromY; rectV[2]==width; rectV[3]==height            
                cv2.rectangle(frame, (rectV[0], rectV[1]), (rectV[0] + rectV[2], rectV[1] + rectV[3]  ), (0, 255, 0), 2)  
          
            _, jpeg_frame = cv2.imencode('.jpg', frame)
            bytes_frame = jpeg_frame.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + bytes_frame + b'\r\n\r\n')

@app.route('/surveillance')
def surveillance():
    return render_template('surveillance.html')

@app.route('/video_viewer', methods=['GET', 'POST'])
def video_viewer():
    global p
    global rectangle
    if request.method == 'POST':
        json = request.get_json()
        mode = json['mode']
        if mode == "on":
            if p is None or not p.is_alive():
                print("[INFO] starting process...")
                time.sleep(0.5)
                p = Process(target=classify_frame, args=(input_queue, output_queue,rectangle,))
                p.daemon = True
                p.start()
            return Response(process_frame(video_stream, input_queue, output_queue, detections, rectangle),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            p.terminate()
            return '', 204
    else:        
        return Response(process_frame(video_stream, input_queue, output_queue, detections, rectangle),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

def classify_frame(input_queue, output_queue, rectangle):
    first_frame = None
    while True:
        # check to see if there is a frame in our input queue
        if not input_queue.empty():
            frame = input_queue.get()
            if frame is not None:
                if not rectangle.empty():
                    rectV = rectangle.get() #rectV-current value of the rectangle
                    rectangle.put(rectV) 
                    #rectV[0]==pixelFromX; rectV[1]==pixelFromY; rectV[2]==width; rectV[3]==height  
                    frame = frame[rectV[0]:rectV[0]+rectV[2],rectV[1]:rectV[1]+rectV[3] ]
                # resize the frame, convert it to grayscale, and blur it
                #frame = imutils.resize(frame, width=500)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                # if the first frame is None, initialize it
                if first_frame is None:
                    first_frame = gray
                    continue
                if len(frame)!=len(first_frame):
                    continue
                # compute the absolute difference between the current frame and first frame
                frame_delta = cv2.absdiff(first_frame, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

                # dilate the thresholded image to fill in holes, then find contours on thresholded image
                thresh = cv2.dilate(thresh, None, iterations=2)
                detections = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                detections = imutils.grab_contours(detections)
                output_queue.put(detections)

def process_frame(cam, input_queue, output_queue, detections, rectangle):
    while True:        
        try:
            if pi:
                frame = cam.read()
            else:
                _, frame = cam.read()                
        except:                
            #One frame wasn't read properly
            pass

        if frame is not None:     

            if input_queue.empty():
                input_queue.put(frame)

            if not output_queue.empty():
                detections = output_queue.get()
            
            if not rectangle.empty():                
                rectV = rectangle.get() #rectV-current value of the rectangle
                rectangle.put(rectV)
                #rectV[0]==pixelFromX; rectV[1]==pixelFromY; rectV[2]==width; rectV[3]==height
                cv2.rectangle(frame, (rectV[0], rectV[1]), (rectV[0] + rectV[2], rectV[1] + rectV[3] ), (0, 0, 255), 2) 
            
            if detections is not None:
                for d in detections:
                    #if cv2.contourArea(d) > 500:
                        # compute the bounding box for the contour, draw it on the frame
                        (x, y, w, h) = cv2.boundingRect(d)
                        if not rectangle.empty():
                            rectVal = rectangle.get()
                            rectangle.put(rectVal)
                            cv2.rectangle(frame, (x+rectVal[0], y+rectVal[1]), (x+rectVal[0] + w, y+rectVal[0] + h), (0, 255, 0), 2)
                        #else:    
                            #cv2.rectangle(frame, (x, y), (x - int(w), y - int(h)), (0, 255, 0), 2)
            
            _, jpeg_frame = cv2.imencode('.jpg', frame)
            bytes_frame = jpeg_frame.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + bytes_frame + b'\r\n\r\n')

if __name__ == '__main__':
    input_queue = Queue(maxsize=1)
    output_queue = Queue(maxsize=1)    
    detections = None
    p = None
    rectangle = Queue(maxsize=1)
    rectangle.put([0, 0, 0, 0])

    print("[INFO] starting video stream...")
    video_stream = VideoStream(src=0).start() if pi else cv2.VideoCapture(0)
    time.sleep(2.0)
    app.run(host='0.0.0.0')
