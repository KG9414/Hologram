# objloader.py

class OBJ:
    def __init__(self, filename):
        self.vertices = []
        self.faces = []

        with open(filename, 'r') as file:
            for line in file:
                tokens = line.strip().split()
                if not tokens:
                    continue

                if tokens[0] == 'v':
                    vertex = list(map(float, tokens[1:]))
                    self.vertices.append(vertex)
                elif tokens[0] == 'f':
                    face = [int(vertex.split('/')[0]) - 1 for vertex in tokens[1:]]
                    self.faces.append(face)
