import argparse
import json
import math
import os
import sys
from math import radians
from shutil import copyfile
import bpy
from PIL import Image
import mathutils

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--model_id', type=str, default='', help='model id to be processed')
parser.add_argument('--dataset', type=str, default='', help='dataset name')
argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)


def clear_scene():
    '''
    clear blender system for scene and object with pose, refer to 3D-R2N2
    '''
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_pattern(pattern="RotCenter")
    bpy.ops.object.select_pattern(pattern="Lights*")
    bpy.ops.object.select_pattern(pattern="Camera")
    bpy.ops.object.select_all(action='INVERT')
    bpy.ops.object.delete()

    # The meshes still present after delete
    for item in bpy.data.lights:
        bpy.data.lights.remove(item)
    for item in bpy.data.cameras:
        bpy.data.cameras.remove(item)
    for item in bpy.data.images:
        bpy.data.images.remove(item)
    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item)
    for item in bpy.data.materials:
        bpy.data.materials.remove(item)


def add_camera(xyz=(0, 0, 0), fov=1.0, name=None, proj_model='PERSP', sensor_fit='HORIZONTAL'):
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object

    cam.rotation_euler[0] = radians(0)
    cam.rotation_euler[1] = radians(0)
    cam.rotation_euler[2] = radians(0)

    if name is not None:
        cam.name = name

    cam.location = xyz
    cam.data.type = proj_model
    cam.data.angle = fov
    cam.data.sensor_fit = sensor_fit

    return cam


def setup_render_all():
    global normal_file_output
    global silhoutte_file_output
    global albedo_file_output

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    vl = bpy.context.view_layer
    vl.use_pass_normal = True
    vl.use_pass_diffuse_color = True
    vl.use_pass_diffuse_direct = True
    vl.use_pass_diffuse_indirect = True
    vl.use_pass_environment = True

    for n in tree.nodes:
        tree.nodes.remove(n)

    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    # depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    # depth_file_output.label = 'Depth Output'
    # links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])

    scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
    scale_normal.blend_type = 'MULTIPLY'
    ##scale_normal.use_alpha = True
    scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])

    bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
    bias_normal.blend_type = 'ADD'
    ##bias_normal.use_alpha = True
    bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_normal.outputs[0], bias_normal.inputs[1])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    links.new(bias_normal.outputs[0], normal_file_output.inputs[0])

    silhoutte_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    silhoutte_file_output.label = 'Silhoutte Output'
    links.new(render_layers.outputs['Env'], silhoutte_file_output.inputs[0])

    albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = 'Albedo Output'
    links.new(render_layers.outputs['DiffCol'], albedo_file_output.inputs[0])


def setup_render_albedo_only():
    global albedo_file_output
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    vl = bpy.context.view_layer
    vl.use_pass_normal = False
    vl.use_pass_diffuse_color = True
    vl.use_pass_diffuse_direct = False
    vl.use_pass_diffuse_indirect = False
    vl.use_pass_environment = False

    for n in tree.nodes:
        tree.nodes.remove(n)

    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = 'Albedo Output'
    links.new(render_layers.outputs['DiffCol'], albedo_file_output.inputs[0])


normal_file_output = silhoutte_file_output = albedo_file_output = None
clear_scene()

scene = bpy.context.scene
scene.render.resolution_x = 256  # default 1200
scene.render.resolution_y = 256  # default 1200
scene.render.resolution_percentage = 100
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.image_settings.color_depth = '8'

radius = 2.25

#phis = list(range(0, 360, 45))
#thetas = list(range(25, 90, 30))

phis = list(range(0, 360, 60))
thetas = list(range(25, 85, 30))


def add_light(name, location):
    # Create light datablock
    light_data = bpy.data.lights.new(name=name + '-data', type='POINT')
    light_data.energy = 100
    # Create new object, pass the light data 
    light_object = bpy.data.objects.new(name=name, object_data=light_data)
    # Link object to collection in context
    bpy.context.collection.objects.link(light_object)
    # Change light position
    light_object.location = location


def look_at(obj_camera, point):
    loc_camera = obj_camera.location
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    print(rot_quat.to_euler())
    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()


def render_pose_function(model_file, texture_file):
    setup_render_all()
    model_id = model_file.split('/')[-2]

    # loading model
    try:
        bpy.ops.import_scene.obj(filepath=model_file)
    except Exception as _err:
        return None

    # tranform object by pose
    # transfrom_object(trans_vec, rot_mat, scale=1.0, name='obj')
    # camera and lighting config
    fov = 1.0462120316141805
    camera = add_camera((0, 0, 0), fov, 'camera')
    for o in bpy.data.objects:
        if o.type == 'CAMERA':
            bpy.context.scene.camera = o
            break

    for output_node in [silhoutte_file_output, normal_file_output, albedo_file_output]:  # , depth_file_output
        output_node.base_path = ''

    model = bpy.context.scene.objects["normalized_model"]  # Get the object
    bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
    bpy.context.view_layer.objects.active = model  # Make the cube the active object
    model.select_set(True)

    world = bpy.data.worlds['World']
    world.light_settings.use_ambient_occlusion = True
    world.light_settings.ao_factor = 1.0

    # set lighting
    add_light('Point', (0, 2.0, 0.5))
    add_light('Point.001', (1.73, -1.0, 0.5))
    add_light('Point.002', (-1.73, -1.0, 0.5))

    uv_mat = bpy.data.materials.new('UVTexture')
    bpy.data.materials['UVTexture'].use_nodes = True
    texture_tree = bpy.data.materials['UVTexture'].node_tree
    texture_links = texture_tree.links
    texture_node = texture_tree.nodes.new("ShaderNodeTexImage")
    texture_node.image = bpy.data.images.load(texture_file)
    model.data.materials[0] = uv_mat
    texture_links.new(texture_node.outputs[0], texture_tree.nodes['Principled BSDF'].inputs[0])
    # bpy.data.scenes['Scene'].render.layers['RenderLayer'].material_override = bpy.data.materials['UVTexture']

    normal_map_name = "NormalMap"
    normal_map = bpy.data.images.new(normal_map_name, texture_node.image.size[0], texture_node.image.size[1])

    for mat in model.data.materials:
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        texture_node = nodes.new('ShaderNodeTexImage')
        texture_node.name = 'Bake_node'
        texture_node.select = True
        nodes.active = texture_node
        texture_node.image = normal_map

    bpy.context.view_layer.objects.active = model
    bpy.context.scene.render.bake.margin = 1
    bpy.ops.object.bake(type='NORMAL', save_mode='EXTERNAL', normal_space="OBJECT")
    normal_map.save_render(os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'surface_normals.png'))

    for mat in model.data.materials:
        for node in mat.node_tree.nodes:
            if node.name == 'Bake_node':
                mat.node_tree.nodes.remove(node)

    # rx, ry, rz = [90, 0, 0]
    # camera.rotation_mode = 'XYZ'
    # camera.rotation_euler[0] = radians(rx)
    # camera.rotation_euler[1] = radians(ry)
    # camera.rotation_euler[2] = radians(rz)

    # bpy.context.scene.render.image_settings.file_format = 'JPEG'
    # bpy.context.scene.render.image_settings.color_depth = '8'

    scene = bpy.context.scene

    ctr = 0
    for theta in thetas:
        for phi in phis:
            camera.location = [radius * math.cos(radians(phi)) * math.sin(radians(theta)), radius * math.sin(radians(phi)) * math.sin(radians(theta)), radius * math.cos(radians(theta))]
            look_at(camera, mathutils.Vector((0, 0, 0)))
            # rendering
            scene.render.filepath = os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'rgb_{ctr:03d}')
            # depth_file_output.file_slots[0].path = os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, 'depth')
            normal_file_output.file_slots[0].path = os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'normal_{ctr:03d}')
            silhoutte_file_output.file_slots[0].path = os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'silhoutte_{ctr:03d}')
            albedo_file_output.file_slots[0].path = os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'albedo_{ctr:03d}')
            bpy.ops.render.render(write_still=True)  # render still
            normal_path = os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'normal_{ctr:03d}0001.png')
            if os.path.exists(normal_path):
                os.rename(normal_path, os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'normal_{ctr:03d}.png'))
            env_path = os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'silhoutte_{ctr:03d}0001.png')
            if os.path.exists(env_path):
                os.rename(env_path, os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'silhoutte_{ctr:03d}.png'))
            albedo_path = os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'albedo_{ctr:03d}0001.png')
            if os.path.exists(albedo_path):
                os.rename(albedo_path, os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'albedo_{ctr:03d}.png'))
            with open(os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'camera_{ctr:03d}.txt'), "w") as fptr:
                fptr.write(json.dumps({
                    'fov': fov,
                    'focal_length_mm': camera.data.lens,
                    't': [camera.location[0], camera.location[1], camera.location[2]],
                    'r': [camera.rotation_euler[0], camera.rotation_euler[1], camera.rotation_euler[2]],
                    'r_mode': camera.rotation_mode
                }))
            ctr += 1

    copyfile(texture_file, os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'surface_texture.png'))
    # delete objects in sys
    # clear_scene()


def render_noc(model_file, texture_file):
    setup_render_albedo_only()
    try:
        bpy.ops.import_scene.obj(filepath=model_file)
    except:
        return None

    fov = 1.0462120316141805
    camera = add_camera((0, 0, 0), fov, 'camera')
    for o in bpy.data.objects:
        if o.type == 'CAMERA':
            bpy.context.scene.camera = o
            break

    model = bpy.context.scene.objects["normalized_model"]  # Get the object
    bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
    bpy.context.view_layer.objects.active = model  # Make the cube the active object
    model.select_set(True)

    for output_node in [albedo_file_output]:  # , depth_file_output
        output_node.base_path = ''

    world = bpy.data.worlds['World']
    world.light_settings.use_ambient_occlusion = True
    world.light_settings.ao_factor = 1.0

    # set lighting
    add_light('Point', (0, 2.0, 0.5))
    add_light('Point.001', (1.73, -1.0, 0.5))
    add_light('Point.002', (-1.73, -1.0, 0.5))

    tex_mat = bpy.data.materials.new('texture_coords')
    bpy.data.materials['texture_coords'].use_nodes = True
    texture_tree = bpy.data.materials['texture_coords'].node_tree
    texture_links = texture_tree.links
    texture_node = texture_tree.nodes.new("ShaderNodeTexCoord")
    model.data.materials[0] = tex_mat
    texture_links.new(texture_node.outputs[0], texture_tree.nodes['Principled BSDF'].inputs[0])
    noc_map_name = "NOCMap"
    image_size = Image.open(texture_file).size
    noc_map = bpy.data.images.new(noc_map_name, image_size[0], image_size[1])

    for mat in model.data.materials:
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        texture_node = nodes.new('ShaderNodeTexImage')
        texture_node.name = 'Bake_node'
        texture_node.select = True
        nodes.active = texture_node
        texture_node.image = noc_map

    model_id = model_file.split('/')[-2]
    bpy.context.view_layer.objects.active = model
    bpy.context.scene.render.bake.margin = 1
    bpy.ops.object.bake(type='DIFFUSE', pass_filter={"COLOR"}, save_mode='EXTERNAL')
    noc_map.save_render(os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'noc.png'))

    for mat in model.data.materials:
        for n in mat.node_tree.nodes:
            if n.name == 'Bake_node':
                mat.node_tree.nodes.remove(n)

    ctr = 0
    for theta in thetas:
        for phi in phis:
            camera.location = [radius * math.cos(radians(phi)) * math.sin(radians(theta)), radius * math.sin(radians(phi)) * math.sin(radians(theta)), radius * math.cos(radians(theta))]
            look_at(camera, mathutils.Vector((0, 0, 0)))
            # rendering
            scene.render.filepath = os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'del_{ctr:03d}')
            albedo_file_output.file_slots[0].path = os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'noc_render_{ctr:03d}')
            bpy.ops.render.render(write_still=True)  # render still
            os.remove(os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'del_{ctr:03d}.png'))
            noc_render_path = os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'noc_render_{ctr:03d}0001.png')
            if os.path.exists(noc_render_path):
                os.rename(noc_render_path, os.path.join(base_dir, f"data/{dataset_name_0}/{dataset_name_1}", model_id, f'noc_render_{ctr:03d}.png'))
            ctr += 1


dataset_name_0, dataset_name_1 = args.dataset.split('/')
_model_file = os.path.join(base_dir, f"data/{dataset_name_0}-model/{dataset_name_1}", args.model_id, "normalized_model.obj")
_texture_file = os.path.join(base_dir, f"data/{dataset_name_0}-model/{dataset_name_1}", args.model_id, "texture.png")

render_pose_function(_model_file, _texture_file)
clear_scene()
render_noc(_model_file, _texture_file)
