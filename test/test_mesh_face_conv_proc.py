from data_processing.process_mesh_for_conv import *


def test_mesh_face_conv_padding():
    mesh = trimesh.load("test/meshes/quad_cube_min.obj", process=False, force='mesh')
    # face_neighbors, vertices, faces, is_pad_vertex, is_pad_face = get_face_neighbors(mesh)
    face_neighbors, vertices, faces, is_pad_vertex, is_pad_face = quadface_8_neighbors(mesh)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export("test_cube.obj")

    mesh = trimesh.load("test/meshes/quad_plane_xz.obj", process=False, force='mesh')
    # face_neighbors, vertices, faces, is_pad_vertex, is_pad_face = get_face_neighbors(mesh)
    face_neighbors, vertices, faces, is_pad_vertex, is_pad_face = quadface_8_neighbors(mesh)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export("test_plane.obj")


def inspect_mesh_triface_neighbors():
    mesh = trimesh.load("test/meshes/tri_plane_xz.obj", process=False, force='mesh')
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


def inspect_mesh_quadface_neighbors():
    mesh = trimesh.load("test/meshes/quad_plane_xz.obj", process=False, force='mesh')
    face_neighbors, vertices, faces, is_pad_vertex, is_pad_face = quadface_8_neighbors(mesh)
    face_neighbors = cartesian_ordering(face_neighbors, faces, vertices)
    print('vertices')
    print(vertices)
    print('faces')
    print(faces)
    print('neighbors')
    print(face_neighbors)
    white, red, green, blue = [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]
    for i in range(len(face_neighbors)):
        colors = np.ones((vertices.shape[0] + 4 * 9, 3), dtype=np.uint8) * 128
        v_i = vertices[faces[i], :]
        f_i = np.array([0, 1, 2, 3])
        colors[0: 4, :] = white
        v, f = [], []
        for n in range(8):
            v_ = vertices[faces[face_neighbors[i][n]], :]
            f_ = np.array([0, 1, 2, 3]) + 4 * (n + 1)
            colors[4 * (n + 1): 4 * (n + 2), :] = np.array([red, green, blue][n % 3]) * [1, 0.6, 0.2][n // 3]
            v.append(v_)
            f.append(f_)
        trimesh.Trimesh(vertices=np.row_stack([v_i] + v + [vertices, ]), faces=np.row_stack([f_i, ] + f + [9 * 4 + faces, ]), vertex_colors=colors, validate=False, process=False).export(f"neighbor_{i:03}.obj")


if __name__ == "__main__":
    # test_mesh_face_conv_padding()
    inspect_mesh_quadface_neighbors()
