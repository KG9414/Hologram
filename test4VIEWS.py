import pyglet
from pyglet.gl import *
from pywavefront import Wavefront

from pyglet.graphics import vertex_list

def load_obj(filename):
    obj = Wavefront(filename)
    vertices = []
    faces = []

    for name, material, group, vertices, normals, texture_coords in obj:
        faces.extend(group)

    return vertices, faces


# Initialize Pyglet
window = pyglet.window.Window(800, 600, resizable=True)
pyglet.gl.glClearColor(1, 1, 1, 1)
glEnable(GL_DEPTH_TEST)

# Load your OBJ file
obj_vertices, obj_faces = load_obj("Metulj.obj")  # Replace with your OBJ file path

@window.event
def on_draw():
    window.clear()
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, window.width / window.height, 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    for i, angle in enumerate(range(0, 360, 90)):
        glPushMatrix()
        glRotatef(angle, 0, 1, 0)

        glBegin(GL_TRIANGLES)
        for face in obj_faces:
            for vertex in face:
                glVertex3fv(obj_vertices[vertex - 1])
        glEnd()

        glPopMatrix()

pyglet.app.run()
