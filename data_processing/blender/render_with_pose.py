import bpy
import pdb
import random
import argparse, sys, os, time
import logging
import numpy as np
import json
from mathutils import Matrix
import math
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from math import radians
import mathutils
from shutil import copyfile

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--model_id', type=str, default='', help='model id to be processed')
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


def add_camera(xyz=(0,0,0), fov=1, name=None, proj_model='PERSP', sensor_fit='HORIZONTAL'):
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

clear_scene()

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

#depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
#depth_file_output.label = 'Depth Output'
#links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])

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

albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
albedo_file_output.label = 'Albedo Output'
links.new(render_layers.outputs['Env'], albedo_file_output.inputs[0])

scene = bpy.context.scene
scene.render.resolution_x = 256    # default 1200
scene.render.resolution_y = 256    # default 1200
scene.render.resolution_percentage = 100

# Delete default cube
for obj in bpy.data.objects:
    obj.select_set(True)
bpy.ops.object.delete()

def add_light(name, location):
    # Create light datablock
    light_data = bpy.data.lights.new(name=name+'-data', type='POINT')
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
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_depth = '8'
    model_id = model_file.split('/')[-2]

    # loading model
    try: 
        bpy.ops.import_scene.obj(filepath=model_file)
    except: return None
    
    # tranform object by pose
    # transfrom_object(trans_vec, rot_mat, scale=1.0, name='obj')
    # camera and lighting config
    fov = 1.0462120316141805
    camera = add_camera((0, 0, 0), fov, 'camera')
    for o in bpy.data.objects:
        if o.type == 'CAMERA':
            bpy.context.scene.camera = o
            break
    
    for output_node in [albedo_file_output, normal_file_output]: #, depth_file_output
        output_node.base_path = ''
    
    model = bpy.context.scene.objects["normalized_model"]       # Get the object
    bpy.ops.object.select_all(action='DESELECT') # Deselect all objects
    bpy.context.view_layer.objects.active = model   # Make the cube the active object 
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
    #bpy.data.scenes['Scene'].render.layers['RenderLayer'].material_override = bpy.data.materials['UVTexture']

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
    normal_map.save_render(os.path.join(base_dir, "data/3D-FUTURE/Sofa", model_id, f'surface_normals.png'))
    
    for mat in model.data.materials:
        for n in mat.node_tree.nodes:
            if n.name == 'Bake_node':
                mat.node_tree.nodes.remove(n)
    
    #rx, ry, rz = [90, 0, 0]
    #camera.rotation_mode = 'XYZ'
    #camera.rotation_euler[0] = radians(rx)
    #camera.rotation_euler[1] = radians(ry)
    #camera.rotation_euler[2] = radians(rz)
    
    #bpy.context.scene.render.image_settings.file_format = 'JPEG'
    #bpy.context.scene.render.image_settings.color_depth = '8'
    
    scene = bpy.context.scene
    
    radius = 2.25
    
    phis = list(range(0, 360, 45))
    thetas = list(range(25, 90, 30))
    
    ctr = 0
    for theta in thetas:
        for phi in phis:
            camera.location = [radius * math.cos(radians(phi)) * math.sin(radians(theta)), radius * math.sin(radians(phi)) * math.sin(radians(theta)), radius * math.cos(radians(theta))]
            look_at(camera, mathutils.Vector((0, 0, 0)))
            # rendering
            scene.render.filepath = os.path.join(base_dir, "data/3D-FUTURE/Sofa", model_id, f'rgb_{ctr:03d}')
            #depth_file_output.file_slots[0].path = os.path.join(base_dir, "data/3D-FUTURE/Sofa", model_id, 'depth')
            normal_file_output.file_slots[0].path = os.path.join(base_dir, "data/3D-FUTURE/Sofa", model_id, f'normal_{ctr:03d}')
            albedo_file_output.file_slots[0].path = os.path.join(base_dir, "data/3D-FUTURE/Sofa", model_id, f'silhoutte_{ctr:03d}')
            bpy.ops.render.render(write_still=True)  # render still
            normal_path = os.path.join(base_dir, "data/3D-FUTURE/Sofa", model_id, f'normal_{ctr:03d}0001.png')
            if os.path.exists(normal_path):
                os.rename(normal_path, os.path.join(base_dir, "data/3D-FUTURE/Sofa", model_id, f'normal_{ctr:03d}.png'))
            env_path = os.path.join(base_dir, "data/3D-FUTURE/Sofa", model_id, f'silhoutte_{ctr:03d}0001.png')
            if os.path.exists(env_path):
                os.rename(env_path, os.path.join(base_dir, "data/3D-FUTURE/Sofa", model_id, f'silhoutte_{ctr:03d}.png'))
            with open(os.path.join(base_dir, "data/3D-FUTURE/Sofa", model_id, f'camera_{ctr:03d}.txt'), "w") as fptr:
                fptr.write(json.dumps({
                    'fov': fov,
                    'focal_length_mm': camera.data.lens,
                    't': [camera.location[0], camera.location[1], camera.location[2]],
                    'r': [camera.rotation_euler[0], camera.rotation_euler[1], camera.rotation_euler[2]],
                    'r_mode': camera.rotation_mode
                }))
            ctr += 1
    
    copyfile(texture_file, os.path.join(base_dir, "data/3D-FUTURE/Sofa", model_id, f'surface_texture.png'))
    # delete objects in sys
    # clear_scene()

model_file = os.path.join(base_dir, "data/3D-FUTURE-model/Sofa", args.model_id, "normalized_model.obj") 
texture_file = os.path.join(base_dir, "data/3D-FUTURE-model/Sofa", args.model_id, "texture.png") 

render_pose_function(model_file, texture_file)