from diffusers import AutoPipelineForInpainting
from diffusers.utils import make_image_grid
from PIL import Image, ImageFilter, ImageOps
import numpy as np


pipeline = AutoPipelineForInpainting.from_pretrained(
    "data/trained_model/inpaint"
    # 'data/pretrained_model',safety_checker=None
)
pipeline.enable_model_cpu_offload()


im = Image.open('data/texture/smpl.png')
im_mask = Image.open('data/mask/smpl_inpaint_mask.png')


# creates a new texture by filling some of black pixels
im_eroded = im.filter(ImageFilter.MaxFilter(3))
im = Image.composite(im, im_eroded, ImageOps.invert(im_mask))

# updates the mask for the new filled texture
im_eroded_np = np.array(im_eroded)
updated_mask_np = (im_eroded_np[:, :, 0] < 2) & (im_eroded_np[:, :, 1] < 2) & (im_eroded_np[:, :, 1] < 2)
im_mask = Image.fromarray(updated_mask_np)

# erodes the mask (to shrink the texture area, aims to get rid of artefacts at the border)
im_mask = im_mask.convert("L").filter(ImageFilter.MaxFilter(5)).convert("P")




prompt = "a smpl texturemap"
# negative_prompt = "bad anatomy, deformed, ugly, disfigured"
image = pipeline(prompt=prompt, image=im, mask_image=im_mask, guidance_scale=3, num_inference_steps=100).images[0]

make_image_grid([im, im_mask, image], rows=1, cols=3).save('data/predict/out_inp.png')

# del pipeline # if use another pipeline below
