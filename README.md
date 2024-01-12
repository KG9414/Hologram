# Interative hologram with gesture detection

## Installation

**For the program to work please install the following pyhton libraries:**

- pip install numpy

- pip install opencv-python

- pip install mediapipe

- pip install pygame PyOpenGL

  

**To run the program:**
python hologram_guestures.py 2> /dev/null

After completing these steps, you can see a camera capturing your gestures alongside a holographic projection of a butterfly in the background. Our system incorporates three gesture detection functionalities for your exploration: swiping to rotate the 3D model, a pinch gesture for zooming in or out, and a rock and roll sign to initiate rapid color changes in the model. The recognized gestures are also displayed in your terminal for your reference. 
Please be aware that the current gesture detection algorithm processes only one distinct gesture per second. However, you have the flexibility to modify this threshold as needed. Additionally, please consider adjusting the gesture detection thresholds to align with your camera's resolution for optimal performance, as the current settings are calibrated for our specific resolution.