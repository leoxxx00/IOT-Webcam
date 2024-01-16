import cv2
import socket
import struct
import pickle
import datetime
import time
import logging

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_fullbody.xml')

detection = False
timer_started = False
detection_stopped_time = 0  # Initialize detection_stopped_time
SECONDS_TO_RECORD_AFTER_DETECTION = 5

frame_size = (640, 480)  # Assuming a standard frame size
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None  # Initialize the VideoWriter variable

def detect_and_record(frame):
    global detection, timer_started, out, detection_stopped_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) + len(bodies) > 0:
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            filename = f"{current_time}.mp4"
            out = cv2.VideoWriter(filename, fourcc, 20.0, frame_size)
            logging.info("Started recording! Filename: %s", filename)
    elif detection:
        if not timer_started:
            timer_started = True
            detection_stopped_time = time.time()
        elif time.time() - detection_stopped_time > SECONDS_TO_RECORD_AFTER_DETECTION:
            detection = False
            timer_started = False
            out.release()
            logging.info('Stop Recording!')

    if detection:
        out.write(frame)

def start_client():
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    server_address = ('localhost', 12345) #replace local host with the ip address of a resberry pi
    client_socket.connect(server_address)
    logging.info(f"Connected to {server_address}")

    # Open a window to display the received video
    cv2.namedWindow("Client Video")

    try:
        while True:
            # Receive the frame size from the server
            message_size = client_socket.recv(struct.calcsize("L"))
            if not message_size:
                break

            message_size = struct.unpack("L", message_size)[0]

            # Receive the frame data from the server
            data = b""
            while len(data) < message_size:
                packet = client_socket.recv(message_size - len(data))
                if not packet:
                    break
                data += packet

            # Deserialize the frame
            frame = pickle.loads(data)

            # Display the received frame
            cv2.imshow("Client Video", frame)
            detect_and_record(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up the connection and close the window
        client_socket.close()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_client()
