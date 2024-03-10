import cv2
import numpy as np
import os

# Function to load multiple template images from a directory
def load_templates(directory):
    templates = []
    for filename in os.listdir(directory):
        #if filename.endswith(".jpeg") or filename.endswith(".jpg"):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as needed
            template_path = os.path.join(directory, filename)
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            templates.append(template)
    return templates

# Load multiple template images from a directory
template_directory = "/Users/abhudaysingh/Documents/templates_Volleyball"  # Specify the directory containing template images
templates = load_templates(template_directory)

# Load the volleyball match video
video = cv2.VideoCapture("/Users/abhudaysingh/Downloads/volleyball_match.mp4")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output.avi', fourcc, 30.0, (int(video.get(3)), int(video.get(4))))

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Resize the frame to a smaller size for better computation
    resized_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)  # Adjust the scaling factor as needed

    # Convert resized frame to grayscale
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    for template in templates:
        # Resize the template to match the resized frame
        resized_template = cv2.resize(template, None, fx=0.5, fy=0.5)  # Adjust the scaling factor as needed

        # Perform template matching with the current template image
        res = cv2.matchTemplate(gray, resized_template, cv2.TM_CCOEFF_NORMED)

        # Define a threshold for the match score
        threshold = 0.79

        # Get the location of the match
        loc = np.where(res >= threshold)

        # Draw a rectangle around the matched region
        for pt in zip(*loc[::-1]):
            cv2.rectangle(resized_frame, pt, (pt[0] + resized_template.shape[1], pt[1] + resized_template.shape[0]), (0, 255, 0), 2)

    # Write the frame to the output video
    output_video.write(resized_frame)

    # Display the frame
    cv2.imshow('Volleyball Tracking', resized_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video objects
video.release()
output_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
