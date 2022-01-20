from PIL import Image
from pathlib import Path
import numpy as np
import cv2 as cv
from tqdm import tqdm
from skimage import color


def concat_masks(path):
    if (path / "mask.png").exists():
        mask = Image.open(path / "mask.png")
        assert mask.size[0] == mask.size[1], f"{str(path)} size_mask"
        mask = mask.resize((256, 256))
        mask.save(path / "images" / "mask.png")
        return
    mask = np.ones((256, 256), dtype=np.bool)
    for impath in ["images/substance_map_minc_vgg16.map.png", "images/substance_map_minc_vgg16.map.v2.png"]:
        try:
            image = Image.open(path / impath)
        except Exception as err:
            print(err)
            continue
        assert image.size[0] == image.size[1], f"{str(path)} size"
        image = np.array(image.resize((256, 256), resample=Image.NEAREST))
        unique_max = np.unique(image).max()
        mask = np.logical_and(mask, image == unique_max)
    mask = np.logical_not(mask)
    Image.fromarray(mask).save(path / "images" / "mask.png")


def dilate_erode_mask(mask_dir):
    (mask_dir.parent / f'{mask_dir.name}_eroded').mkdir(exist_ok=True)
    for mask_path in tqdm(list(mask_dir.iterdir())):
        mask = np.array(Image.open(mask_path)).astype(np.uint8)
        kernel_size = 2
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1), (kernel_size, kernel_size))
        mask = cv.erode(mask, element)
        mask = mask > 0
        Image.fromarray(mask).save(mask_dir.parent / f'{mask_dir.name}_eroded' / mask_path.name)


def cv_segment(src_path, dst_path):
    try:
        image = Image.open(src_path)
        orig_arr = np.array(image)
        outlined_arr = np.array(image)
        outlined_arr[cv.Canny(outlined_arr, 10, 200) > 0, :] = 0
        lab_orig_arr = color.rgb2lab(orig_arr)
        sure_fg = lab_orig_arr[:, :, 0] < 95
        sure_bg = lab_orig_arr[:, :, 0] > 99
        gc_mask = np.ones((outlined_arr.shape[0], outlined_arr.shape[1]), dtype=np.uint8) * cv.GC_PR_BGD
        gc_mask[sure_fg] = cv.GC_FGD
        gc_mask[sure_bg] = cv.GC_BGD
        fgmodel = np.zeros((1, 65), dtype=np.float)
        bgmodel = np.zeros((1, 65), dtype=np.float)
        mask, _, _ = cv.grabCut(outlined_arr, gc_mask, None, bgmodel, fgmodel, iterCount=10, mode=cv.GC_INIT_WITH_MASK)
        Image.fromarray(np.logical_or(mask == cv.GC_PR_FGD, mask == cv.GC_FGD)).save(dst_path)
    except Exception as err:
        print('Error:', err, src_path)


def smooth_mask(src_path, dst_path):
    image = np.array(Image.open(src_path))
    blur = cv.blur(image, (21 * 2, 21 * 2))
    Image.fromarray(blur).save(dst_path)


def create_segmentations():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str)
    parser.add_argument('-m', '--mask_folder', type=str)
    parser.add_argument('-o', '--output_folder', type=str)
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)

    args = parser.parse_args()

    files = sorted([x for x in Path(args.input_folder).iterdir()])
    # files = [x for i, x in enumerate(files) if i % args.num_proc == args.proc and x.name == 'shape02317_rank02_pair35114.jpg']

    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True)

    for f in tqdm(files):
        cv_segment(f, output_folder / f.name)
        # smooth_mask(Path(args.mask_folder) / f.name, Path(args.mask_folder).parent / f'{Path(args.mask_folder).name}_soft' / f.name)


if __name__ == '__main__':
    create_segmentations()
    # dilate_erode_mask(Path('/cluster/gimli/ysiddiqui/CADTextures/Photoshape/exemplars_mask'))
