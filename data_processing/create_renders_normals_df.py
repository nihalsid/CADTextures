from argparse import ArgumentParser
import subprocess

from util.misc import read_list, write_list

blender_render_cmd = lambda blender, dataset, model_id: f"{blender} --background --python data_processing/blender/render_with_pose.py -- --model_id {model_id} --dataset {dataset}"
sdf_gen_cmd = lambda dataset, model_id: f"python data_processing/sdf_gen/process_meshes.py --model_id {model_id} --dataset {dataset}"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='3D-FUTURE/Sofa', type=str)
    parser.add_argument('--blender', default='', type=str)
    parser.add_argument('--num_proc', default=1, type=int)
    parser.add_argument('--proc', default=0, type=int)
    args = parser.parse_args()

    shapes_all = read_list(f'data/splits/{args.dataset}/official/all.txt')
    shapes = [x for i, x in enumerate(shapes_all) if i % args.num_proc == args.proc]
    # shapes = shapes[:2]
    failed_shapes = []
    for shape in shapes:
        try:
            subprocess.call(blender_render_cmd(args.blender, args.dataset, shape), shell=True)
            subprocess.call(sdf_gen_cmd(args.dataset, shape), shell=True)
        except Exception as err:
            print("ERROR:", err)
            failed_shapes.append(shape)
    if len(failed_shapes) > 0:
        write_list("failed_shapes.txt", failed_shapes)
