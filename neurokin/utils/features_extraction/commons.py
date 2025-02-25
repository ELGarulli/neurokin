import numpy as np
def angle(vectors):
    if vectors.shape[1] == 9:
        a, b, c = vectors[:, :3], vectors[:, 3:6], vectors[:, 6:9]
    elif vectors.shape[1] == 6:
        a, b, c = vectors[:, 2], vectors[:, 2:4], vectors[:, 4:6]

    bas = a - b
    bcs = c - b
    angles = []
    for ba, bc in zip(bas, bcs):
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        angles.append(angle)
    return np.array(angles)