# Security camera based on embedded system
##### Raspberry Pi with OpenCV and Flask
When motion is detected the led is on and the object is framed on screen.
![picamera](https://user-images.githubusercontent.com/33002299/51426473-b20d6e80-1beb-11e9-9171-69041998a68e.jpg)
#### Run
Get python3 and all necessary libraries and call `python3 motion_detection.py`
Go to the provided localhost address, configure detection area and press "System start".
Face detection is also available but with less functional server.
Running the application on computer is possible after removing lines associated with GPIO managing.
