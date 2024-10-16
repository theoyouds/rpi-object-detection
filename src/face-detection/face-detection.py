import cv2
import numpy as np
import time

from picamera2 import Picamera2

# ------------------------------------------------------------------------------
# automaticdai
# YF Robotics Labrotary
# Instagram: yfrobotics
# Twitter: @yfrobotics
# Website: https://www.yfrl.org
# ------------------------------------------------------------------------------
# Reference:
# - https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
# ------------------------------------------------------------------------------

fps = 0

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240


def visualize_fps(image, fps: int):
    if len(np.shape(image)) < 3:
        text_color = (255, 255, 255)  # white
    else:
        text_color = (0, 255, 0)  # green
    row_size = 20  # pixels
    left_margin = 24  # pixels

    font_size = 1
    font_thickness = 1

    # Draw the FPS counter
    fps_text = "FPS = {:.1f}".format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(
        image,
        fps_text,
        text_location,
        cv2.FONT_HERSHEY_PLAIN,
        font_size,
        text_color,
        font_thickness,
    )

    return image


def CalculateVector(
    x: int, y: int, w: int, h: int, img_width: int, img_height: int
) -> tuple[float, float]:
    frame_x_centre = img_width / 2
    frame_y_centre = img_height / 2

    face_x_centre = x + w / 2
    face_y_centre = y + h / 2

    vector = (face_x_centre - frame_x_centre, face_y_centre - frame_y_centre)

    return vector


# Load the cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

picam2 = Picamera2()
picam2.start()

while True:
    # ----------------------------------------------------------------------
    # record start time
    start_time = time.time()
    # Read the frame
    img = picam2.capture_array()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    max_face_area = 0
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if (w * h) > max_face_area:
            max_face_area = w * h
            vector = CalculateVector(x, y, w, h, np.shape(img)[1], np.shape(img)[0])
            print(f"Vector: {vector}")
            centre_point = (np.shape(img)[1] / 2, np.shape(img)[0] / 2)
            print(f"Centre point: {centre_point}")
            end_point = (centre_point[0] + vector[0], centre_point[1] + vector[1])
            print(f"End point: {end_point}")
            # cv2.line(img, centre_point, end_point, (0, 255, 0), 2)

    # Display
    cv2.imshow("img", visualize_fps(img, fps))
    # ----------------------------------------------------------------------
    # record end time
    end_time = time.time()
    # calculate FPS
    seconds = end_time - start_time
    fps = 1.0 / seconds
    print("Estimated fps:{0:0.1f}".format(fps))
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
