from pathlib import Path
from PIL import Image, ImageOps, ImageFont
import numpy as np
from PIL import ImageDraw
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from pygifsicle import optimize
from tqdm import tqdm


def write_text_to_image(array, text):
    img = Image.fromarray(array)
    img = ImageOps.expand(img, (0, 20, 0, 0), fill=(255, 255, 255))
    width, height = img.size
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 14)
    draw.text((2, 10), text, (0, 0, 0), font=font)
    return np.array(img)


def create_run_timelapse(run):
    Path(run, "timelapse").mkdir(exist_ok=True)
    epochs = sorted([x for x in (Path(run) / "visualization").iterdir() if x.name.startswith('epoch_')])[:-1]
    for fig_path in tqdm(list((epochs[0] / 'val_vis').iterdir())):
        collection = []
        for e in epochs:
            collection.append(write_text_to_image(np.array(Image.open(e / 'val_vis' / fig_path.name)), e.name))
        clip = ImageSequenceClip(collection, fps=2)
        opath = Path(run, "timelapse", '.'.join(fig_path.name.split('.')[:-1]) + '.gif')
        clip.write_gif(opath, verbose=False, logger=None)
        # optimize(str(opath), options=["--no-warnings"])


if __name__ == "__main__":
    _run = "/media/nihalsid/OSDisk/Users/ga83fiz/nihalsid/CADTextures/runs/20071701_end2end_fast_dev/"
    create_run_timelapse(_run)
