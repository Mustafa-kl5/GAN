import cv2
import os
import numpy as np
from mtcnn import MTCNN

# Parameters
video_path = 'test.mp4'  # Path to the video file
output_dir = 'extracted_faces'         # Directory to save the extracted faces
img_shape = (32, 32, 3)                # Size of the output images
frame_interval = 5                     # Process every 5th frame to speed up

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the MTCNN face detector
detector = MTCNN()

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_count = 0
face_count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Process every nth frame (frame_interval)
    if frame_count % frame_interval != 0:
        continue

    # Detect faces in the frame
    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']

        # Extract the face from the frame
        face_img = frame[y:y+h, x:x+w]

        # Resize the face to the desired size
        face_resized = cv2.resize(face_img, (img_shape[0], img_shape[1]))

        # Save the face image
        face_filename = os.path.join(output_dir, f'face_{frame_count}_{face_count}.png')
        cv2.imwrite(face_filename, face_resized)
        face_count += 1

    print(f'Processed frame {frame_count}, found {len(faces)} faces.')

# Release the video capture object
cap.release()

print(f'Extraction completed. {face_count} faces were saved to "{output_dir}".')
