import cv2

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
hat=cv2.imread('C:/Users/kusha/PycharmProjects/Filters_cv/Filters/hat.png')
glass=cv2.imread('C:/Users/kusha/PycharmProjects/Filters_cv/Filters/glasses.png')
dog=cv2.imread('C:/Users/kusha/PycharmProjects/Filters_cv/Filters/dog.png')

def put_dog_filter(dog, fc, x, y, w, h):
    face_width = w
    face_height = h

    dog = cv2.resize(dog, (int(face_width * 1.5), int(face_height * 1.95)))
    for i in range(int(face_height * 1.75)):
        for j in range(int(face_width * 1.5)):
            for k in range(3):
                if dog[i][j][k] < 235:
                    fc[y + i - int(0.375 * h) - 1][x + j - int(0.35 * w)][k] = dog[i][j][k]
    return fc

def put_hat(hat, fc, x, y, w, h):
    face_width = w
    face_height = h

    hat_width = face_width + 1
    hat_height = int(0.50 * face_height) + 1

    hat = cv2.resize(hat, (hat_width, hat_height))

    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if hat[i][j][k] < 235:
                    fc[y + i - int(0.40 * face_height)][x + j][k] = hat[i][j][k]
    return fc


def put_glass(glass, fc, x, y, w, h):
    face_width = w
    face_height = h

    hat_width = face_width + 1
    hat_height = int(0.50 * face_height) + 1

    glass = cv2.resize(glass, (hat_width, hat_height))

    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if glass[i][j][k] < 235:
                    fc[y + i - int(-0.20 * face_height)][x + j][k] = glass[i][j][k]
    return fc
global choise

choice = 0
print('enter your choice filter to launch that: 1="put hat & glasses" ,any number="put fog filters" ')
choise= int(input('enter your choice:'))
webcam = cv2.VideoCapture(0)
while True:
    size=4
    (rval, im) = webcam.read()
    im = cv2.flip(im, 1, 0)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    fl = face.detectMultiScale(gray,1.19,7)

    for (x, y, w, h) in fl:
        if choise ==1:
            im = put_hat(hat, im, x, y, w, h)
            im = put_glass(glass, im, x, y, w, h)

        else:
            im = put_dog_filter(dog, im, x, y, w, h)

    cv2.imshow('Hat & glasses',im)
    key = cv2.waitKey(30) & 0xff
    if key == 27:  # The Esc key
       break


# This code is a simple implementation of a real-time face filter application using OpenCV, a computer vision library. Here's a breakdown of what the code does:

# Importing Libraries and Loading Assets

# The code starts by importing the OpenCV library (cv2) and loading three images: hat.png, glasses.png, and dog.png. These images will be used as filters to be applied to the user's face.

# Defining Functions

# The code defines three functions:

# put_dog_filter: This function takes in the dog image, the frame (fc), and the coordinates of the face (x, y, w, h). It resizes the dog image to fit the face, and then overlays the dog image onto the frame, replacing the pixels in the frame with the corresponding pixels from the dog image.
# put_hat: This function takes in the hat image, the frame (fc), and the coordinates of the face (x, y, w, h). It resizes the hat image to fit the face, and then overlays the hat image onto the frame, replacing the pixels in the frame with the corresponding pixels from the hat image.
# put_glass: This function is similar to put_hat, but it overlays the glasses image onto the frame instead.
# Main Loop

# The code then enters a main loop, where it:

# Reads a frame from the webcam using cv2.VideoCapture.
# Flips the frame horizontally using cv2.flip.
# Converts the frame to grayscale using cv2.cvtColor.
# Detects faces in the grayscale frame using face.detectMultiScale.
# For each detected face, it applies the chosen filter (either put_hat and put_glass, or put_dog_filter) to the frame.
# Displays the resulting frame using cv2.imshow.
# Waits for a key press using cv2.waitKey. If the user presses the Esc key (keycode 27), the loop exits.
# User Input

# Before entering the main loop, the code prompts the user to enter a choice of filter to apply. If the user enters 1, the code applies the put_hat and put_glass filters. Otherwise, it applies the put_dog_filter.

# Overall, this code provides a simple example of how to use OpenCV to detect faces and apply filters to a live video feed.



## The concept behind

# Let's break down the concept of how the code applies a filter to a face in a live video feed:

# Face Detection

# The code uses the face.detectMultiScale function from OpenCV to detect faces in the grayscale frame. This function uses a pre-trained classifier (in this case, haarcascade_frontalface_default.xml) to identify regions of the image that resemble a face.

# Here's a simplified overview of how face detection works:

# The classifier is trained on a dataset of images that contain faces, as well as images that do not contain faces.
# The classifier learns to identify patterns and features that are common to faces, such as the shape of the eyes, nose, and mouth.
# When the classifier is applied to a new image, it scans the image for regions that match the patterns and features it has learned.
# If a region matches the patterns and features, the classifier returns a bounding box around the region, indicating that a face has been detected.
# Filter Application

# Once a face has been detected, the code applies a filter to the face by overlaying an image (e.g. a hat or dog ears) onto the original frame. Here's a simplified overview of how this works:

# The code resizes the filter image to match the size of the detected face.
# The code then overlays the filter image onto the original frame, using the bounding box coordinates returned by the face detector to position the filter correctly.
# To overlay the filter image, the code uses a technique called alpha blending, which combines the pixels of the filter image with the pixels of the original frame.
# Alpha blending works by assigning an alpha value to each pixel in the filter image, which determines how transparent or opaque the pixel should be.
# When the filter image is overlaid onto the original frame, the alpha values are used to blend the pixels of the two images together, creating a seamless composite image.
# Pixel-Level Operations

# At the pixel level, the code performs the following operations to apply the filter:

# For each pixel in the filter image, the code checks the alpha value to determine how transparent or opaque the pixel should be.
# If the alpha value is high (i.e. the pixel is opaque), the code replaces the corresponding pixel in the original frame with the pixel from the filter image.
# If the alpha value is low (i.e. the pixel is transparent), the code leaves the corresponding pixel in the original frame unchanged.
# By performing these pixel-level operations for each pixel in the filter image, the code creates a seamless composite image that combines the filter with the original frame.
# Overall, the code uses a combination of face detection, image resizing, and alpha blending to apply a filter to a face in a live video feed.