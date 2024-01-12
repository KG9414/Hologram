import cv2
import time
import mediapipe as mp
import numpy as np
import pygame
import random
import copy
import math
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import gluPerspective
from objloader import OBJ

# python hologram_guestures.py 2> /dev/null
# python3 hologram_guestures.py 2> /dev/null


# pip install numpy
# pip install opencv-python
# pip3 install mediapipe
# pip install pygame PyOpenGL

def load_model(model_path):
    obj = OBJ(model_path)
    return obj


def mirror_model_x(model):
    mirrored_model = OBJ(model.Metulj.obj)
    mirrored_model.vertices = np.copy(model.vertices)
    mirrored_model.vertices[:, 0] *= -1
    mirrored_model.faces = model.faces
    return mirrored_model


class GestureDetector:
    def __init__(self, max_frames=5, cooldown_duration=1):
        self.last_hand_landmarks = []
        self.max_frames = max_frames
        self.last_pinch_distance = None
        self.last_gesture_time = time.time()  # Initialize last gesture time
        self.cooldown_duration = cooldown_duration

    def detect_zoom_gesture(self):
        if len(self.last_hand_landmarks) < 2:
            return None

        x_coordinates = [
            hand_landmark.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x for hand_landmark in self.last_hand_landmarks]
        total_index_distance = np.diff(x_coordinates).sum()



        swipe_threshold_left = 0.15
        swipe_threshold_right = -0.15

        thumb_tip = self.last_hand_landmarks[-1].landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        index_tip = self.last_hand_landmarks[-1].landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

        wrist = self.last_hand_landmarks[-1].landmark[mp.solutions.hands.HandLandmark.WRIST]
        middle_finger = self.last_hand_landmarks[-1].landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]

        # Calculate the distance between the thumb and index finger tips, linalg.norm is eucledean norm + wrist middle finger distnce
        pinch_distance = np.linalg.norm(
            np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
        wmf_distance = np.linalg.norm(
            np.array([wrist.x, wrist.y]) - np.array([middle_finger.x, middle_finger.y]))

        # prvic
        if self.last_pinch_distance is None:
            self.last_pinch_distance = pinch_distance
            return None

        # Calculate the pinch change - negativna 3 - 10 -> squeesh/zoom out    pozitivna -> grem narazen/zoom in
        pinch_change = pinch_distance - self.last_pinch_distance

        zoom_threshold_in = 0.02
        zoom_threshold_out = 0.05

        current_time = time.time()
        # ce je ze over cooldown duration + ce je tud middle_finger_wrist distance manjsa od 0.15 (mamo skrceno pest)
        if pinch_change > 0.07 and current_time - self.last_gesture_time > self.cooldown_duration and wmf_distance < 0.15:  # poz, razpon
            self.last_gesture_time = current_time
            self.last_pinch_distance = pinch_distance
            return "Zoom In"

        elif pinch_change < -0.07 and current_time - self.last_gesture_time > self.cooldown_duration and pinch_distance < 0.30 and wmf_distance < 0.15:  # neg, squeesh
            self.last_gesture_time = current_time
            self.last_pinch_distance = pinch_distance
            return "Zoom Out"

        self.last_pinch_distance = pinch_distance
        return None

    def detect_swipe_gesture(self):
        if len(self.last_hand_landmarks) < 2:
            return None

        x_coordinates = [
            hand_landmark.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x for hand_landmark in self.last_hand_landmarks]

        thumb_tip = self.last_hand_landmarks[-1].landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        index_tip = self.last_hand_landmarks[-1].landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        wrist = self.last_hand_landmarks[-1].landmark[mp.solutions.hands.HandLandmark.WRIST]
        middle_finger = self.last_hand_landmarks[-1].landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]

        wmf_distance = np.linalg.norm(
            np.array([wrist.x, wrist.y]) - np.array([middle_finger.x, middle_finger.y]))

        total_index_distance = np.diff(x_coordinates).sum()
        swipe_threshold_left = 0.15
        swipe_threshold_right = -0.15

        current_time = time.time()

        if total_index_distance > swipe_threshold_left and wmf_distance < 0.20 and current_time - self.last_gesture_time > self.cooldown_duration:
            self.last_gesture_time = current_time
            return "left"
        elif total_index_distance < swipe_threshold_right and wmf_distance < 0.20 and current_time - self.last_gesture_time > self.cooldown_duration:
            self.last_gesture_time = current_time
            return "right"

        return None

    def detect_rock_on_gesture(self):
        if len(self.last_hand_landmarks) < 2:
            return None

        index_tip = self.last_hand_landmarks[-1].landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        pinky_tip = self.last_hand_landmarks[-1].landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
        middle_tip = self.last_hand_landmarks[-1].landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = self.last_hand_landmarks[-1].landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
        thumb_tip = self.last_hand_landmarks[-1].landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]

        vertical_distance = abs(index_tip.y - pinky_tip.y)

        middle_distance = abs(index_tip.y - middle_tip.y)
        ring_distance = abs(index_tip.y - ring_tip.y)
        thumb_distance = abs(index_tip.y - thumb_tip.y)

        rock_on_threshold = 0.04

        if (
            vertical_distance < rock_on_threshold and
            middle_distance > rock_on_threshold and
            ring_distance > rock_on_threshold and
            thumb_distance > rock_on_threshold
        ):
            return "Rock On"

        return None

    def reset_gesture_state(self):
        self.last_hand_landmarks = []


def draw_cross():
    glLineWidth(3.0)

    glDisable(GL_LIGHTING)

    glBegin(GL_LINES)
    glColor3f(1.0, 1.0, 1.0)
    glVertex2f(-1.0, 0.0)
    glVertex2f(1.0, 0.0)
    glVertex2f(0.0, -1.0)
    glVertex2f(0.0, 1.0)
    glEnd()

    glEnable(GL_LIGHTING)



def draw_model(model, position, rotation, scale, color):

    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    glRotatef(rotation[0], 1, 0, 0)  # Rotate around the X-axis
    glRotatef(rotation[1], 0, 1, 0)  # Rotate around the Y-axis
    glRotatef(rotation[2], 0, 0, 1)  # Rotate around the Z-axis
    glScalef(scale[0], scale[1], scale[2])  # Apply scaling

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color)

    glDisable(GL_LIGHTING)

    glEnable(GL_POINT_SMOOTH)
    glPointSize(1.0)
    glBegin(GL_POINTS)
    glColor4f(*color)
    for vertex_id in range(len(model.vertices)):
        glVertex3fv(model.vertices[vertex_id])
    glEnd()

    glEnable(GL_LIGHTING)
    glPopMatrix()


def draw_model2(model, position, rotation, scale, color, transformation_matrix):
    glPushMatrix()
    glMultMatrixf(transformation_matrix)

    glTranslatef(*position)
    glRotatef(rotation[0], 1, 0, 0)
    glRotatef(rotation[1], 0, 1, 0)
    glRotatef(rotation[2], 0, 0, 1)
    glScalef(*scale)

    glDisable(GL_LIGHTING)
    glEnable(GL_POINT_SMOOTH)
    glPointSize(1.0)
    glBegin(GL_POINTS)
    glColor4f(*color)

    for vertex_id in range(len(model.vertices)):
        glVertex3fv(model.vertices[vertex_id])
    glEnd()
    glEnable(GL_LIGHTING)

    glPopMatrix()


def show_camera():
    global global_rotation
    global_rotation = ""
    cap = cv2.VideoCapture(0)
    gesture_detector = GestureDetector()

    pygame.init()
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    rotation_angle = 0
    zoom = 0.23
    novGesture = False
    global_zoom = ""
    previous_global_zoom = ""

    start_time = time.time()
    display_duration = 5.0  # 5 seconds duration for displaying the cross

    current_model_color = (1.0, 0.5, 1.0, 1.0)

    color_change_duration = 5.0
    color_change_start_time = None
    color_change_color = (1.0, 1.0, 1.0, 1.0)  # Red color

    model_matrix = np.eye(4)

    while cap.isOpened():
        zac = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        rock_on_gesture = gesture_detector.detect_rock_on_gesture()
        if rock_on_gesture:
            print("ROCK ON!")

            color_change_start_time = time.time()
            color_change_color = (
                random.random(), random.random(), random.random(), 1.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        rotation_speed = 12

        if global_rotation == "left":
            rotation_angle -= rotation_speed
        elif global_rotation == "right":
            rotation_angle += rotation_speed

        glRotatef(rotation_angle, 0, 1, 0)
        if global_zoom == "Zoom In" and (novGesture == True):
            zoom += 0.05
            novGesture = False
        elif global_zoom == "Zoom Out" and (novGesture == True):
            zoom -= 0.05
            novGesture = False

        rock_on_gesture = gesture_detector.detect_rock_on_gesture()
        if rock_on_gesture:
            print("ROCK ON!")

            current_model_color = (
                random.random(), random.random(), random.random(), 1.0)

        if color_change_start_time is not None:
            elapsed_color_change_time = time.time() - color_change_start_time
            if elapsed_color_change_time < color_change_duration:
                current_model_color = color_change_color
            else:
                color_change_start_time = None

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        elapsed_time = time.time() - start_time

        # Set the viewport for the entire display - zato da ni kriz cez enga mejhnega
        if elapsed_time < display_duration:
            glViewport(0, 0, display[0], display[1])

            draw_cross()

        for i, viewport in enumerate(viewports):
            glViewport(*viewport)
            scale_factor = 0.7

            model_matrix = np.eye(4)
            glMultMatrixf(model_matrix)

            draw_model2(my_model, butterfly_positions[i], butterfly_rotations[i], (
                zoom, zoom, zoom), current_model_color, model_matrix)

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
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                    if len(gesture_detector.last_hand_landmarks) == gesture_detector.max_frames:
                        gesture_detector.last_hand_landmarks.pop(0)

                    gesture_detector.last_hand_landmarks.append(landmarks)

                    if len(gesture_detector.last_hand_landmarks) == gesture_detector.max_frames:
                        swipe_gesture = gesture_detector.detect_swipe_gesture()
                        zoom_gesture = gesture_detector.detect_zoom_gesture()
                        # rock_on_gesture = gesture_detector.detect_rock_on_gesture()

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
                            print("Zoom Gesture:", zoom_gesture, "\n")
                            gesture_detector.reset_gesture_state()
                        # elif rock_on_gesture:
                        #     print("ROCK ON!")
                        #     glColor4f(1.0, 1.0, 1.0, 1.0)
                        #     gesture_detector.reset_gesture_state()

            cv2.imshow("Camera Feed", frame)

            if not results.multi_hand_landmarks:
                gesture_detector.reset_gesture_state()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "metulj.obj"
    display = (1024, 700)
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

    pygame.init()

    screen_info = pygame.display.Info()

    display = (screen_info.current_w, screen_info.current_h)

    # ZAKOMENTIRI TOLE LAJNO CE HOCES FULL SCREEN
    # display = (400,400)

    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    viewports = [
        (0, 0, display[0] // 2, display[1] // 2),  # Top-left view
        (display[0] // 2, 0, display[0] // 2,
         display[1] // 2),  # Top-right view
        (0, display[1] // 2, display[0] // 2,
         display[1] // 2),  # Bottom-left view
        (display[0] // 2, display[1] // 2, display[0] //
         2, display[1] // 2),  # Bottom-right view
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
    # viewports = [(0, 0, display[0], display[1])]
    show_camera()
