#!/usr/bin/env python3
'''
img2tex program.
'''
import os
import torch
import subprocess
from PIL import Image
from pathlib import Path
from tqdm import tqdm

torch.set_default_device('cuda' if torch.cuda.is_available()
                         else 'mps' if torch.backends.mps.is_available()
                         else 'cpu')
device = torch.get_default_device()

def _preprocess():
    '''
    Convert all input images into .png and delete the original.
    '''
    if not os.path.exists(os.path.join('data', 'input')):
        print("Input folder doesn't exist.")
        exit(1)

    image_paths = list(Path(os.path.join('data', 'input')).iterdir())

    # Get all images that need conversion
    convert_image_paths = []
    for i in range(len(image_paths)):
        if image_paths[i].suffix != '.png':
            convert_image_paths.append(image_paths[i])
            if os.path.exists(image_paths[i].with_suffix('.png')):
                print('Possible overwrite of ', str(image_paths[i].with_suffix('.png')))
                print('Please remove it from the input folder')
                exit(1)

    # Convert .png and delete other images
    for image_path in tqdm(convert_image_paths, total=len(convert_image_paths), desc='Preprocess'):
        Image.open(image_path).save(image_path.with_suffix('.png'))
        os.remove(image_path)


def _img2mat():
    '''
    1. Preprocess input images.
    2. Create human mats from input images.
    '''
    HUMAN_MAT = {
        'input': os.path.join('data', 'input'),
        'output': os.path.join('data', 'human_mat'),
        'weight': os.path.join('data', 'human_mat', 'pretrained_weight', 'SGHM-ResNet50.pth')
    }

    if not os.path.exists(os.path.join('data', 'human_mat')):
        os.mkdir(os.path.join('data', 'human_mat'))

    ##
    # Preprocess input images
    ##
    _preprocess()
    
    ##
    # Create human mats
    ##
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(os.path.dirname(__file__), 'SemanticGuidedHumanMatting')
    cmd = ['python', os.path.join('SemanticGuidedHumanMatting', 'test_image.py'), '--images-dir', HUMAN_MAT['input'], '--result-dir', HUMAN_MAT['output'], '--pretrained-weight', HUMAN_MAT['weight']]
    
    subprocess.run(cmd, capture_output=True, env=env)


def _img2densepose():
    '''
    1. Preprocess input images.
    2. Create human mats from input images.
    3. Create densepose data for input images.
    4. Create densepose images from densepose data.
    5. Mask the densepose images with the human mats.
    '''
    DENSEPOSE = {
        'config': os.path.join('detectron2', 'projects', 'DensePose', 'configs', 'densepose_rcnn_R_101_FPN_s1x.yaml'),
        'model': 'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/165712084/model_final_c6ab63.pkl',
        'input': os.path.join('data', 'input'),
        'output': os.path.join('data', 'densepose', 'densepose.pkl')
    }

    if not os.path.exists(os.path.join('data', 'densepose')):
        os.mkdir(os.path.join('data', 'densepose'))
    
    ##
    # Preprocess input images and create human mats
    ##
    _img2mat()

    ##
    # Create densepose data
    ##
    cmd = ['python', os.path.join('detectron2', 'projects', 'DensePose', 'apply_net.py'), 'dump', DENSEPOSE['config'], DENSEPOSE['model'], DENSEPOSE['input'], '--output', DENSEPOSE['output']]

    subprocess.run(cmd, capture_output=True)

    ##
    # Create densepose images
    ##
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'detectron2', 'projects', 'DensePose'))
    with open(DENSEPOSE['output'], 'rb') as f:
        data = torch.load(f, weights_only=False)

    # Iterate on densepose data
    from torchvision.io import decode_image, write_png
    from torchvision.transforms.v2.functional import pad
    for j in tqdm(range(len(data)), total=len(data), desc='DensePose'):
        # Get filename
        filename = Path(data[j]['file_name']).name

        # Get i,u,v for only the first prediction (not for multiple humans)
        i = data[j]['pred_densepose'][0].labels # torch.long in [0, 24] (bg + 24ps)
        u, v = data[j]['pred_densepose'][0].uv[(0, 1), :, :] # torch.float in (0, 1)
        v = 1 - v # reverse v coordinates
        iuv = torch.stack([i, u * 255, v * 255]).to(torch.uint8)
        
        # Get input image
        img = decode_image(os.path.join(DENSEPOSE['input'], filename)) # (C,H,W)
        float_margins = data[j]["pred_boxes_XYXY"][0]
        margins = torch.round(float_margins).int()
        # Pad: left, top, right, bottom
        padding = [margins[0], img.size(1) - margins[3], img.size(2) - margins[2], margins[1]]

        # Tweak padding to perfectly fit iuv to the input image
        h_diff = margins[3] - margins[1] - iuv.size(1)
        w_diff = margins[2] - margins[0] - iuv.size(2)
        padding[0 if float_margins[0] % 1 >= float_margins[2] % 1 else 2] += w_diff
        padding[1 if float_margins[1] % 1 >= float_margins[3] % 1 else 3] += h_diff
        
        # Pad iuv with 0s to match the input image
        iuv = pad(iuv, padding)
        assert iuv.size() == img.size()

        # Mask the densepose image
        mat = decode_image(os.path.join('data', 'human_mat', filename)).to(device) # (1,H,W)
        mat = mat.to(torch.float32) / 255
        iuv *= (mat >= 0.5)

        # Mask the background where i == 0
        iuv[(iuv[0,:,:] == 0).unsqueeze(0).expand([3, -1, -1])] = 0

        # Save the densepose image
        write_png(iuv.cpu(), os.path.join('data', 'densepose', filename))


def img2texture():
    '''
    1. Preprocess input images.
    2. Create human mats from input images.
    3. Create densepose data for input images.
    4. Create densepose images from densepose data.
    5. Mask the densepose images with the human mats.
    6. Create partial SMPL textures from the densepose and input images.
    '''
    ATLAS_PART_SIZE = 128
    SMPL_PART_SIZE = 512 # match stable-diffusion-2-inpainting resolution

    if not os.path.exists(os.path.join('data', 'smpl_texture')):
        os.mkdir(os.path.join('data', 'smpl_texture'))

    ##
    # Create masked densepose
    ##
    _img2densepose()

    # Construct warp kernel for atlas textures
    import warp as wp
    @wp.kernel
    def img2atlas(image: wp.array2d(dtype=wp.vec3i), # type: ignore
                iuv_data: wp.array3d(dtype=wp.int32), # type: ignore
                texture: wp.array3d(dtype=wp.vec3i)): # type: ignore
        tid_h, tid_w = wp.tid()

        if iuv_data[0, tid_h, tid_w] != 0:
            texture[iuv_data[0, tid_h, tid_w] - 1,
                    iuv_data[1, tid_h, tid_w],
                    iuv_data[2, tid_h, tid_w]] = image[tid_h, tid_w]

    import numpy as np
    from torchvision.io import decode_image
    from UVTextureConverter import Atlas2Normal
    converter = Atlas2Normal(atlas_size=ATLAS_PART_SIZE, normal_size=SMPL_PART_SIZE)
    
    # Iterate on input images
    image_paths = list(Path(os.path.join('data', 'input')).iterdir())
    for j in tqdm(range(len(image_paths)), total=len(image_paths), desc='Texture'):
        filename = image_paths[j].name
        img = decode_image(image_paths[j]).to(device)
        iuv = decode_image(os.path.join('data', 'densepose', filename)).to(device)
        assert iuv.size() == img.size()

        # Convert iuv to index atlas textures
        iuv = torch.vstack([iuv[0].unsqueeze(0), iuv[1:3].float() / 255 * (ATLAS_PART_SIZE - 1)]).int()
        
        # Launch the warp kernel to assign atlas textures
        _, h, w = img.size()
        atlas_textures = wp.zeros((24, ATLAS_PART_SIZE, ATLAS_PART_SIZE), dtype=wp.vec3i)
        img = wp.from_torch(img.permute(1, 2, 0).int(), dtype=wp.vec3i) # (H,W)
        iuv = wp.from_torch(iuv, dtype=wp.int32) # (C,H,W)

        wp.launch(kernel=img2atlas,
                dim=(h, w),
                inputs=[img, iuv, atlas_textures])
        wp.synchronize()

        atlas_textures = wp.to_torch(atlas_textures).to(torch.uint8) # (N,H,W,C)

        ##
        # Convert Atlas texture to SMPL texture
        ##
        smpl_texture = converter.convert(atlas_textures.cpu().numpy())
        smpl_texture = np.uint8((smpl_texture * 255).round()) # (L,L,C)

        # Save SMPL texture
        Image.fromarray(smpl_texture).save(os.path.join('data', 'smpl_texture', filename))
        

def texture2mask():
    '''
    Create inpaint mask (inverted and padded) from texture.
    '''
    if not os.path.exists(os.path.join('data', 'smpl_mask')):
        os.mkdir(os.path.join('data', 'smpl_mask'))

    import numpy as np
    texture_paths = list(Path(os.path.join('data', 'smpl_texture')).iterdir())
    for j in tqdm(range(len(texture_paths)), total=len(texture_paths), desc='Mask'):
        smpl_texture = np.array(Image.open(texture_paths[j]))

        # Create inpaint mask (inverted and padded)
        smpl_mask = (smpl_texture[:,:,0] < 2) & (smpl_texture[:,:,1] < 2) & (smpl_texture[:,:,2] < 2) # (L,L)
        smpl_mask = np.uint8(smpl_mask) * 255
        Image.fromarray(smpl_mask).save(os.path.join('data', 'smpl_mask', texture_paths[j].name))


def train():
    '''
    Finetune stable-diffusion-2-inpainting model with DreamBooth.
    '''
    if not os.path.exists(os.path.join('data', 'trained_model')):
        os.mkdir(os.path.join('data', 'trained_model'))

    import time
    DIFFUSERS = {
        'model': 'stabilityai/stable-diffusion-2-inpainting',
        'input_dir': os.path.join('data', 'training_data', 'image'),
        'mask_dir': os.path.join('data', 'training_data', 'mask'),
        'output_dir': os.path.join('data', 'trained_model', str(int(time.time()))),
        'input_prompt': 'a smpl texturemap',
        'resolution': 512, # all training data will be resized into this resolution
        'max_train_steps': 500 * 3,
        'checkpoint_steps': 500,
        'gradient_accumulation_steps': 2, # larger means more stable update at cost of larger VRAM occupation
        'lr_scheduler': 'constant', # cosine is better than constant
        'learning_rate': 1e-6, # larger g_a_s above means bigger update, so smaller learning_rate is used
        'use_8bit_adam': True,
        'allow_tf32': True, # only True on Ampere+ GPUs
        'mixed_precision': 'fp16'
    }

    cmd = ['accelerate', 'launch', 'train_dreambooth_inpaint.py', '--pretrained_model_name_or_path=' + DIFFUSERS['model'], '--instance_data_dir=' + DIFFUSERS['input_dir'], '--mask_data_dir=' + DIFFUSERS['mask_dir'], '--instance_prompt="' + DIFFUSERS['input_prompt'] + '"', '--output_dir=' + DIFFUSERS['output_dir'], '--resolution=' + str(DIFFUSERS['resolution']), '--train_text_encoder', '--train_batch_size=1', '--max_train_steps=' + str(DIFFUSERS['max_train_steps']), '--checkpointing_steps=' + str(DIFFUSERS['checkpoint_steps']), '--gradient_accumulation_steps=' + str(DIFFUSERS['gradient_accumulation_steps']), '--gradient_checkpointing', '--learning_rate=' + str(DIFFUSERS['learning_rate']), '--lr_scheduler=' + DIFFUSERS['lr_scheduler'], '--lr_warmup_steps=0', '--mixed_precision=' + DIFFUSERS['mixed_precision']]
    if DIFFUSERS['use_8bit_adam']:
        cmd.append('--use_8bit_adam')
    if DIFFUSERS['allow_tf32']:
        cmd.append('--allow_tf32')
    
    subprocess.run(cmd)


def render():
    '''
    Render front images of textured SMPL human model
    '''
    if not os.path.exists(os.path.join('data', 'render')):
        os.mkdir(os.path.join('data', 'render'))

    from pytorch3d.io import load_obj
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        HardPhongShader,
        BlendParams,
        TexturesUV
    )

    # Load object
    verts, faces, aux = load_obj(
        os.path.join('data', 'smpl_model', 'smpl_uv.obj'),
        load_textures=False,
        device=device
    )

    faces_idx = faces.verts_idx # index of each face
    uvs = aux.verts_uvs # uv coordinate of each vertex
    faces_uvs = faces.textures_idx # uv coordinate of each face

    # Set camera
    R, T = look_at_view_transform(1.59, 0, 0, at=[[0, -.31 ,0]], device=device)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    # Set background
    blend_params = BlendParams(background_color=(0,0,0))

    # Set rasterization
    raster_settings = RasterizationSettings(image_size=512)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            cameras=cameras,
            lights=None,
            materials=None,
            blend_params=blend_params,
            device=device
        )
    )

    # Load texture image 512x512
    from torchvision.io import decode_image, write_png
    texture_paths = list(Path(os.path.join('data', 'smpl_texture_inpaint')).iterdir())
    for j in tqdm(range(len(texture_paths)), total=len(texture_paths), desc='Render'):
        texture_image = decode_image(texture_paths[j]).to(device)
        texture_image = (texture_image.to(torch.float32) / 255).permute(1, 2, 0) # (H,W,C)

        # Create texture
        textures = TexturesUV(maps=[texture_image], faces_uvs=[faces_uvs], verts_uvs=[uvs])

        # Create mesh
        mesh = Meshes(verts=[verts], faces=[faces_idx], textures=textures)

        images = renderer(mesh)
        image = images[0, ..., :3].permute(2,0,1) # (C,H,W) remove alpha channel
        image = (image * 255).to(torch.uint8)

        write_png(image.cpu(), os.path.join('data', 'render', texture_paths[j].name))


def inpaint():
    '''
    Inpaint partial SMPL texture with mask.
    '''
    DIFFUSERS = {
        'model': os.path.join('data', 'trained_model', 'inpaint'),
        'prompt': 'a smpl texturemap',
        'guidance_scale': 3,
        'num_inference_steps': 100,
        'use_tf32': True,
        'preprocess': False # interpolate small holes
    }
    if not os.path.exists(os.path.join('data', 'smpl_texture_inpaint')):
        os.mkdir(os.path.join('data', 'smpl_texture_inpaint'))
    
    if DIFFUSERS['use_tf32']:
        torch.backends.cuda.matmul.allow_tf32 = True

    from diffusers import StableDiffusionInpaintPipeline
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        DIFFUSERS['model'],
        torch_dtype=torch.float16,
        # safety_checker=None # bypass NSFW check in sd1.4, not required in sd2+
        use_safetensors=True # for safetensors file
    )
    pipeline.enable_model_cpu_offload()

    import numpy as np
    from PIL import ImageFilter, ImageOps

    texture_paths = list(Path(os.path.join('data', 'smpl_texture')).iterdir())
    for j in tqdm(range(len(texture_paths)), total=len(texture_paths), desc='Inpaint', position=1):
        im = Image.open(texture_paths[j])
        im_np = np.array(im)
        im_mask = (im_np[:,:,0] < 2) & (im_np[:,:,1] < 2) & (im_np[:,:,2] < 2) # (L,L)
        im_mask = Image.fromarray(np.uint8(im_mask) * 255)

        # Preprocess texture and mask to interpolate small holes
        if DIFFUSERS['preprocess']:
            im_eroded = im.filter(ImageFilter.MaxFilter(3))
            im = Image.composite(im, im_eroded, ImageOps.invert(im_mask))

            im_eroded = np.array(im_eroded)
            im_mask = Image.fromarray((im_eroded[:,:,0] < 2) & (im_eroded[:,:,1] < 2) & (im_eroded[:,:,1] < 2))
            im_mask = im_mask.convert("L").filter(ImageFilter.MaxFilter(5)).convert("P")

        image = pipeline(prompt=DIFFUSERS['prompt'], image=im, mask_image=im_mask, guidance_scale=DIFFUSERS['guidance_scale'], num_inference_steps=DIFFUSERS['num_inference_steps']).images[0]

        image.save(os.path.join('data', 'smpl_texture_inpaint', texture_paths[j].name))


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='img2tex')
    parser.add_argument(
        'command',
        nargs=1,
        choices=['i2tex', 'inpaint', 'tex2mask', 'render', 'train'],
        help='[i2tex] images to SMPL textures\n'
        + '[inpaint] inpaint SMPL textures\n'
        + '[tex2mask] SMPL textures to SMPL masks\n'
        + '[render] render SMPL front view\n'
        + '[train] train sd2-inpainting with dreambooth'
    )
    args = parser.parse_args()

    match args.command[0]:
        case 'i2tex':
            img2texture()
        case 'inpaint':
            inpaint()
        case 'tex2mask':
            texture2mask()
        case 'render':
            render()
        case 'train':
            train()
        case _:
            pass

if __name__ == '__main__':
    main()
