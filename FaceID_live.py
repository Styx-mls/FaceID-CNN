import cv2
from PIL import Image
from torchvision import transforms
import torch
from FaceID_classes_and_functions import FaceID_Agent, FaceID_Network
import pickle
from mtcnn import MTCNN

# Load the pre-trained FaceID agent
agent = torch.load("FaceID_Agent.pth")

# Load the face database from a pickle file
with open("face_database.pkl", "rb") as f:
    agent.database = pickle.load(f)

# Set the CNN to evaluation mode for inference
agent.cnn.eval()

# Define preprocessing steps for input face images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.Resize((160, 160)),               # Resize to 160x160
    transforms.ToTensor(),                       # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with mean 0.5 and std 0.5
])

# Initialize webcam for live video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set frame width
cap.set(4, 480)  # Set frame height
cap.set(5, 120)  # Set frame rate

embedding = None  # Placeholder for face embedding
detector = MTCNN()  # Initialize MTCNN for face detection
recognized = False  # Flag to indicate if a face has been recognized

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:  # Exit if no frame is captured
        break

    # Convert the frame from BGR (OpenCV format) to RGB for MTCNN
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        # Get face coordinates and keypoints
        x, y, w, h = face['box']
        keypoints = face['keypoints']

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display key facial landmarks
        for key, point in keypoints.items():
            cv2.circle(frame, point, 2, (0, 0, 255), -1)

        # Extract the region of interest (face)
        face_roi = frame[y:y + h, x:x + w]

        # Convert the face region to a PIL image and preprocess it
        face_image = Image.fromarray(face_roi)
        image_tensor = transform(face_image).unsqueeze(0)

        # Generate an embedding for the detected face using the model
        with torch.no_grad():
            embedding = agent.cnn(image_tensor)

        # Check if the face is recognized
        if agent.recognize_face(embedding) != "Unknown":
            print("Face Recognized, welcome ", agent.recognize_face(embedding))
            recognized = True

    # Display the video feed with annotations
    cv2.imshow("Feed", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Exit the loop if a face has been recognized
    if recognized:
        break


# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
