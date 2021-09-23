from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import os

# setup logging
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('subdivide')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'/tmp/manifold_{datetime.now().strftime("%d%m%H%M%S")}.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


DEPTH = 6


def manifold_meshes(input_folder, target_folder, process_id, total_procs):
    target_folder.mkdir(exist_ok=True, parents=True)
    meshes = sorted([(x / "models" / "model_normalized.obj") for x in input_folder.iterdir() if (x / "models" / "model_normalized.obj").exists()], key=lambda x: x.name)
    # meshes = sorted([(x / "model_normalized.obj") for x in input_folder.iterdir() if (x / "model_normalized.obj").exists()], key=lambda x: x.name)
    meshes = [x for i, x in enumerate(meshes) if i % total_procs == process_id]
    logger.info(f'Proc {process_id + 1}/{total_procs} processing {len(meshes)}')
    for mesh in tqdm(meshes):
        (target_folder / mesh.parents[1].name).mkdir(exist_ok=True)
        # (target_folder / mesh.parents[0].name).mkdir(exist_ok=True)
        dirname = os.path.dirname(os.path.realpath(__file__))
        os.system(f"{dirname}/manifold/manifold --input {mesh} --output {target_folder / mesh.parents[1].name / 'model_normalized.obj'} --depth {DEPTH}")
        # os.system(f"cp {mesh} {target_folder / mesh.parents[0].name / 'model_normalized.obj'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str)
    parser.add_argument('-o', '--output_folder', type=str)
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)

    args = parser.parse_args()
    manifold_meshes(Path(args.input_folder), Path(args.output_folder), args.proc, args.num_proc)
