import socket
import cv2
import pickle
import struct

def start_server():
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to a specific address and port
    server_address = ('localhost', 12345)#replace local host with the ip address of a resberry pi
    server_socket.bind(server_address)

    # Listen for incoming connections
    server_socket.listen(1)
    print(f"Server listening on {server_address}")

    # Wait for a connection
    connection, client_address = server_socket.accept()
    print(f"Connection from {client_address}")

    # Open the webcam
    cap = cv2.VideoCapture(0)

    try:
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()

            # Serialize the frame
            data = pickle.dumps(frame)
            message_size = struct.pack("L", len(data))

            # Send the frame size and frame data to the client
            connection.sendall(message_size + data)

    finally:
        # Clean up the connection and release the webcam
        connection.close()
        cap.release()

if __name__ == "__main__":
    start_server()
