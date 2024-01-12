import cv2
import time
import mediapipe as mp
import numpy as np
import pygame
import random
import copy
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import gluPerspective
from objloader import OBJ

# PETRA S TEM POGANJI PA NE BO TISTIH DODATNIH IZPISOV !!!
#python hologram_guestures.py 2> /dev/null

# pip install numpy
# pip install opencv-python
# pip3 install mediapipe
# pip install pygame PyOpenGL

def load_model(model_path):
    obj = OBJ(model_path)
    return obj

def mirror_model_x(model):
    mirrored_model = OBJ(model.Metulj.obj)  # Pass the filename to the constructor
    mirrored_model.vertices = np.copy(model.vertices)
    mirrored_model.vertices[:, 0] *= -1
    mirrored_model.faces = model.faces
    return mirrored_model


class GestureDetector:
    def __init__(self, max_frames=5, cooldown_duration=0.2):
        self.last_hand_landmarks = []
        self.max_frames = max_frames
        self.last_pinch_distance = None
        self.last_gesture_time = time.time()  # Initialize last gesture time
        self.cooldown_duration = cooldown_duration

    def detect_zoom_gesture(self):
        if len(self.last_hand_landmarks) < 2:
            # print("return < 2", len(self.last_hand_landmarks))
            return None

        # Extract the thumb and index finger tip landmarks
        thumb_tip = self.last_hand_landmarks[-1].landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        index_tip = self.last_hand_landmarks[-1].landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

        # Calculate the distance between the thumb and index finger tips, linalg.norm is eucledean norm
        pinch_distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
    

        # Store the initial pinch distance for reference, first time around
        if self.last_pinch_distance is None:
            self.last_pinch_distance = pinch_distance
            return None


        # Calculate the pinch change - negativna 3 - 10 -> squeesh/zoom out    pozitivna -> grem narazen/zoom in
        pinch_change = pinch_distance - self.last_pinch_distance
        

        # Define a threshold for zoom detection
        zoom_threshold_in = 0.02
        zoom_threshold_out = 0.05 


        # print(pinch_distance, "  ", self.last_pinch_distance)
        
        # print(pinch_change , "\n")

        current_time = time.time()
        if pinch_change > 0.07 and current_time - self.last_gesture_time > self.cooldown_duration: #poz
            # print(pinch_change )
            
            self.last_gesture_time = current_time
            self.last_pinch_distance = pinch_distance
            return "Zoom In"
            # return "razpon"

        elif pinch_change < -0.07 and current_time - self.last_gesture_time > self.cooldown_duration and pinch_distance < 0.15: # neg
            # print(pinch_change)
            self.last_gesture_time = current_time
            self.last_pinch_distance = pinch_distance
            return "Zoom Out"
            # return "squeesh"

    
        self.last_pinch_distance = pinch_distance
        return None

       


    def detect_swipe_gesture(self):
        if len(self.last_hand_landmarks) < 2:
            return None

        # Extract x-coordinates of the index finger tip
        x_coordinates = [hand_landmark.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x for hand_landmark in self.last_hand_landmarks]


        # Calculate total movement in the x-direction
        total_index_distance = np.diff(x_coordinates).sum()
        # Define thresholds for swipe detection
        # Right swipe is associated with an increase in x-values, and a left swipe is associated with a decrease in x-values.
        swipe_threshold_left = 0.15  # Adjust this value based on your hand size and camera resolution
        swipe_threshold_right = -0.15  # Adjust this value based on your hand size and camera resolution

        # print(total_index_distance, swipe_threshold_left )
        # Detect swipe gestures
        if total_index_distance > swipe_threshold_left:
            # print(total_index_distance, "\n")
            return "left"
        elif total_index_distance < swipe_threshold_right:
            # print(total_index_distance,"\n")
            return "right"

        return None
    
    def reset_gesture_state(self):
        self.last_hand_landmarks = []

# def draw_model(model, position, rotation, scale):
def draw_model(model, position, rotation, scale):
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    glRotatef(rotation[0], 1, 0, 0)  # Rotate around the X-axis
    glRotatef(rotation[1], 0, 1, 0)  # Rotate around the Y-axis
    glRotatef(rotation[2], 0, 0, 1)  # Rotate around the Z-axis
    glScalef(scale[0], scale[1], scale[2])  # Apply scaling
    glDisable(GL_LIGHTING)
    glEnable(GL_POINT_SMOOTH)
    glPointSize(1.0)
    glBegin(GL_POINTS)
    glColor4f(1.0, 0.5, 1.0, 1.0)  # Set color to white (R, G, B, Alpha)

    for vertex_id in range(len(model.vertices)):
        glVertex3fv(model.vertices[vertex_id])
    glEnd()

    glEnable(GL_LIGHTING)
    glPopMatrix()


'''
def show_camera():
    # Load your Blender model
    # model_path = "/Users/karlagliha/Documents/Documents/Faks/Magisterij/IOI/SeminarII/FRI_Logo/Metulj.obj"

    cap = cv2.VideoCapture(0)
    prvic = True
    gesture_detector = GestureDetector()

    pygame.init()
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    rotation_angle = 0 
    


    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    global_rotation = ""
    global_zoom = ""
    zoom = 0.26
    novGesture = False
    previous_global_zoom = ""

    while cap.isOpened():
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         quit()

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        rotation_speed = 12  # Adjust the rotation speed as needed
        
        # Apply continuous rotation
        # glRotatef(rotation_angle, 0, 1, 0)
        

        # print("global:",global_rotation)
        # print("rotation_angle:",rotation_angle)


        if global_rotation == "left":
            rotation_angle -= rotation_speed
            glRotatef(rotation_angle, 0, 1, 0)  # Continuous rotation to the left
        elif global_rotation == "right":

            rotation_angle += rotation_speed
            glRotatef(rotation_angle, 0, 1, 0)  # Continuous rotation to the right




        if global_zoom == "Zoom In" and (novGesture == True):
            print(" + 20 ", zoom)
            zoom +=0.1
            novGesture = False
        elif global_zoom == "Zoom Out" and (novGesture == True):
            zoom -=0.1
            print(" - 20 ", zoom)
            novGesture = False
        
        # for light_id in range(GL_LIGHT0, GL_LIGHT2 + 1):
        #     random_color = random.choice(colors_palette)
        #     glLightfv(light_id, GL_DIFFUSE, random_color)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


        for i, viewport in enumerate(viewports):
            glViewport(*viewport)
            scale_factor = 0.7  # You can adjust this value as needed

            draw_model(butterfly_instances[i], butterfly_positions[i], butterfly_rotations[i], (zoom, zoom, zoom))

        pygame.display.flip()
        pygame.time.wait(10)

        with mp.solutions.hands.Hands() as hands:
            glEnable(GL_LIGHTING)
            glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.5, 0.5, 0.5, 1.0))


            # glRotatef(3, 0, 1, 0)





            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    
                    mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                    # If size is > 15, remove the oldest frame if not just add the new one each iteration (first iteration)
                    if len(gesture_detector.last_hand_landmarks) == gesture_detector.max_frames:
                        gesture_detector.last_hand_landmarks.pop(0) # Remove the oldest frame

                    gesture_detector.last_hand_landmarks.append(landmarks)  # Add the newest frame

                    
                    # Check for recognision only if len = 15 (torej vsakih 15 frameov), plus when we detect we reset the whole array
                    if len(gesture_detector.last_hand_landmarks) == gesture_detector.max_frames:
                        # print("iz show camera ",len(gesture_detector.last_hand_landmarks))
                        swipe_gesture = gesture_detector.detect_swipe_gesture()
                        zoom_gesture = gesture_detector.detect_zoom_gesture()
                        novGesture = True
                        global_zoom = zoom_gesture

                        if swipe_gesture:
                            print("Swipe Gesture:", swipe_gesture,"\n")
                            if swipe_gesture == "left":
                                global_rotation = "left"
                            else:
                                global_rotation = "right"
                            gesture_detector.reset_gesture_state()
                        elif zoom_gesture:
                            print("!!!Zoom Gesture:",  zoom_gesture , "\n")
 
                            gesture_detector.reset_gesture_state()
                            
                    


            cv2.imshow("Camera Feed", frame)

            # Reset gesture state if no hand is detected
            if not results.multi_hand_landmarks:
                gesture_detector.reset_gesture_state()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
'''

def show_camera():
    global global_rotation
    global_rotation = ""
    cap = cv2.VideoCapture(0)
    gesture_detector = GestureDetector()

    pygame.init()
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    # Adjust perspective and camera position
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    rotation_angle = 0
    zoom = 0.26
    novGesture = False
    global_zoom = ""
    previous_global_zoom = ""

    while cap.isOpened():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        rotation_speed = 12

        if global_rotation == "left":
            rotation_angle -= rotation_speed
        elif global_rotation == "right":
            rotation_angle += rotation_speed

        glRotatef(rotation_angle, 0, 1, 0)

        if global_zoom == "Zoom In" and (novGesture == True):
            print(" + 20 ", zoom)
            zoom +=0.1
            novGesture = False
        elif global_zoom == "Zoom Out" and (novGesture == True):
            zoom -=0.1
            print(" - 20 ", zoom)
            novGesture = False

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        for i, viewport in enumerate(viewports):
            glViewport(*viewport)
            scale_factor = 0.7

            # Adjust the translation to center the model
            draw_model(butterfly_instances[i], (0.0, 0.0, 0.0), butterfly_rotations[i], (zoom, zoom, zoom))

        pygame.display.flip()
        pygame.time.wait(10)

        with mp.solutions.hands.Hands() as hands:
            glEnable(GL_LIGHTING)
            glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.5, 0.5, 0.5, 1.0))

            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                    if len(gesture_detector.last_hand_landmarks) == gesture_detector.max_frames:
                        gesture_detector.last_hand_landmarks.pop(0)

                    gesture_detector.last_hand_landmarks.append(landmarks)

                    if len(gesture_detector.last_hand_landmarks) == gesture_detector.max_frames:
                        swipe_gesture = gesture_detector.detect_swipe_gesture()
                        zoom_gesture = gesture_detector.detect_zoom_gesture()
                        novGesture = True
                        global_zoom = zoom_gesture

                        if swipe_gesture:
                            print("Swipe Gesture:", swipe_gesture, "\n")
                            if swipe_gesture == "left":
                                global_rotation = "left"
                            else:
                                global_rotation = "right"
                            gesture_detector.reset_gesture_state()
                        elif zoom_gesture:
                            print("!!!Zoom Gesture:", zoom_gesture, "\n")
                            gesture_detector.reset_gesture_state()

            cv2.imshow("Camera Feed", frame)

            if not results.multi_hand_landmarks:
                gesture_detector.reset_gesture_state()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    model_path = "metulj.obj"
    display = (1024,700)
    # display = (1700,1024)
    # 1024px × 1366px


    colors_palette = [
        (1.0, 0.0, 0.0, 1.0),  # Red
        (0.0, 1.0, 0.0, 1.0),  # Green
        (0.0, 0.0, 1.0, 1.0),  # Blue
        (1.0, 1.0, 0.0, 1.0),  # Yellow
        (1.0, 0.0, 1.0, 1.0),  # Magenta
        (0.0, 1.0, 1.0, 1.0),  # Cyan
    ]
    print(1366/4.0)
    butterfly_positions = [
        (0.6, 0.5, 0.0),  # Top-left
        (-0.6, 0.5, 0.0),  # Top-right
        (0.6, -0.5, 0.0),  # Bottom-left
        (-0.6, -0.5, 0.0),  # Bottom-right
    ]

    butterfly_rotations = [
        (0, 0, 135),
        (0, 0, -135),  
        (0, 0, 45),  
        (0, 0, -45),  
    ]

    my_model = load_model(model_path)

    # Initialize pygame and set the display size
    pygame.init()

    screen_info = pygame.display.Info()

    # Set the display size to match the current screen size
    display = (screen_info.current_w, screen_info.current_h)

    # Create the display
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    # pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    # Create four instances of the model
    butterfly_instances = [copy.copy(my_model) for _ in range(4)]

    viewports = [
    (0, 0, display[0] // 2, display[1] // 2),  # Top-left view
    (display[0] // 2, 0, display[0] // 2, display[1] // 2),  # Top-right view
    (0, display[1] // 2, display[0] // 2, display[1] // 2),  # Bottom-left view
    (display[0] // 2, display[1] // 2, display[0] // 2, display[1] // 2),  # Bottom-right view
    ]

    butterfly_positions = [
        (0.0, 0.0, 0.0),      # Top-left
        (0.0, 0.0, 0.0),      # Top-right
        (0.0, 0.0, 0.0),      # Bottom-left
        (0.0, 0.0, 0.0),      # Bottom-right
    ]

    butterfly_rotations = [
        (0, 0, 135),
        (0, 0, -135),
        (0, 0, 45),
        (0, 0, -45),
    ]

    # en layout
    #viewports = [(0, 0, display[0], display[1])]
    show_camera()
