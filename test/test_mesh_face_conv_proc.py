from data_processing.process_mesh_for_conv import *


def test_mesh_face_conv_padding():
    mesh = trimesh.load("test/meshes/cube_min.obj", process=False, force='mesh')
    face_neighbors, vertices, faces, is_pad_vertex, is_pad_face = get_face_neighbors(mesh)
    trimesh.Trimesh(vertices=vertices, faces=faces).export("test_cube.obj")

    mesh = trimesh.load("test/meshes/plane_xy.obj", process=False, force='mesh')
    face_neighbors, vertices, faces, is_pad_vertex, is_pad_face = get_face_neighbors(mesh)
    trimesh.Trimesh(vertices=vertices, faces=faces).export("test_plane.obj")


def inspect_mesh_face_neighbors():
    mesh = trimesh.load("test/meshes/plane_yz.obj", process=False, force='mesh')
    face_neighbors, vertices, faces, is_pad_vertex, is_pad_face = get_face_neighbors(mesh)
    face_neighbors = cartesian_ordering(face_neighbors, faces, vertices)
    print('vertices')
    print(vertices)
    print('faces')
    print(faces)
    print('neighbors')
    print(face_neighbors)
    white, red, green, blue = [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]
    for i in range(len(face_neighbors)):
        colors = np.ones((vertices.shape[0] + 3 * 4, 3), dtype=np.uint8) * 128
        v_i = vertices[faces[i], :]
        f_i = np.array([0, 1, 2])
        colors[0: 3, :] = white

        v_0 = vertices[faces[face_neighbors[i][0]], :]
        f_0 = np.array([0, 1, 2]) + 3
        colors[3: 6, :] = red

        v_1 = vertices[faces[face_neighbors[i][1]], :]
        f_1 = np.array([0, 1, 2]) + 6
        colors[6: 9, :] = green

        v_2 = vertices[faces[face_neighbors[i][2]], :]
        f_2 = np.array([0, 1, 2]) + 9
        colors[9: 12, :] = blue

        trimesh.Trimesh(vertices=np.row_stack((v_i, v_0, v_1, v_2, vertices)), faces=np.row_stack((f_i, f_0, f_1, f_2, 12 + faces)), vertex_colors=colors, validate=False, process=False).export(f"neighbor_{i:03}.obj")


if __name__ == "__main__":
    inspect_mesh_face_neighbors()
