def loadOBJ(filepath: str) -> list:

    file = open(filepath, "r")
    vertices = []
    for line in file:
        vals = line.split()

        if len(vals) == 0:
            continue

        if vals[0] == "v":
            v = list(map(float, vals[1:4]))
            vertices.append(v)

    return vertices
