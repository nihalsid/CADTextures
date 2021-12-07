import bpy
import os
from pathlib import Path


def clear_objects():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_pattern(pattern="RotCenter")
    bpy.ops.object.select_pattern(pattern="Lights*")
    bpy.ops.object.select_pattern(pattern="Camera")
    bpy.ops.object.select_all(action='INVERT')
    bpy.ops.object.delete()
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


def process_file(input_blend, output_dir):
    file_path = str(input_blend)
    inner_path = "Objects"
    object_name = "uvmapped_v2.resized"

    with bpy.data.libraries.load(file_path, link=False) as (data_from, data_to):
        data_to.objects = [name for name in data_from.objects if name.startswith(object_name)]

    print(data_to.objects)

    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)

    for item in bpy.data.images:
        if item.name not in ['Render Result', 'Viewer Node']:
            item.filepath_raw = str(output_dir / os.path.basename(item.filepath_raw))
            item.save()

    bpy.ops.export_scene.obj(filepath=str(output_dir / "model_normalized.obj"))


if __name__ == "__main__":
    root_output_dir = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs")
    blends = [x for x in Path("/cluster_HDD/gondor/ysiddiqui/shapenet-chairs").iterdir() if x.name.endswith(".blend")]
    output_dirs = [root_output_dir / x.name.split('.blend')[0] for x in blends]

    for idx, [blend, out_dir] in enumerate(zip(blends, output_dirs)):
        clear_objects()
        out_dir.mkdir(exist_ok=True, parents=True)
        process_file(blend, out_dir)
        print(f'Done {idx}/{len(blends)}')
