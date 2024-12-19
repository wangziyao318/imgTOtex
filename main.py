#!/usr/bin/env python3
'''
img2tex program entry point.
'''

import time
import os
import torch
from torchvision.io import decode_image, write_png
from torchvision.utils import make_grid
from torchvision.transforms.v2.functional import pad, invert

'''
params
'''
# The program only supports one input image with single human in it
INPUT_IMG = 'sample.jpg'
MASK_THRESHOLD = 250
ATLAS_PART_LEN = 128
SMPL_PART_LEN = 512 # for stable diffusion input

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")



HUMANMATTING = {
    'input': os.path.join('data', 'source'),
    'output': os.path.join('data', 'mask'),
    'weight': os.path.join('data', 'pretrained_weight', 'SGHM-ResNet50.pth')
}

def img2mask():
    '''
    Create human mask without background from the input image.
    The INPUT_DIR should only contain the INPUT_IMG.
    '''
    # Create human mask
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(os.path.dirname(__file__), 'SemanticGuidedHumanMatting')
    cmd = ['python', os.path.join('SemanticGuidedHumanMatting', 'test_image.py'), '--images-dir', HUMANMATTING['input'], '--result-dir', HUMANMATTING['output'], '--pretrained-weight', HUMANMATTING['weight']]
    
    import subprocess
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    stdout, _ = process.communicate() # wait for async popen to complete



DENSEPOSE = {
    'cfg': os.path.join('detectron2', 'projects', 'DensePose', 'configs', 'densepose_rcnn_R_101_FPN_s1x.yaml'),
    'model': 'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/165712084/model_final_c6ab63.pkl',
    'input': os.path.join('data', 'source', INPUT_IMG),
    'output': os.path.join('data', 'densepose', 'densepose.pkl')
}

def img2densepose():
    '''
    Create densepose data for the input image,
    and then create densepose image from densepose data.
    Finally, mask the densepose image with the human mask created in img2mask().
    '''
    # Create densepose data
    cmd = ['python', os.path.join('detectron2', 'projects', 'DensePose', 'apply_net.py'), 'dump', DENSEPOSE['cfg'], DENSEPOSE['model'], DENSEPOSE['input'], '--output', DENSEPOSE['output']]

    import subprocess
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = process.communicate()

    # Create human mask
    img2mask()

    # Create densepose image
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'detectron2', 'projects', 'DensePose'))
    with open(DENSEPOSE['output'], 'rb') as f:
        data = torch.load(f, weights_only=False)

    i = data[0]['pred_densepose'][0].labels # torch.long in 0-24 (background + 24 body parts)
    u, v = data[0]['pred_densepose'][0].uv[(0, 1), :, :] # torch.float in (0,1)
    v = 1 - v # reverse v coordinates
    iuv = torch.stack([i, u * 255, v * 255]).to(torch.uint8)
    
    img = decode_image(DENSEPOSE['input']) # (3,H,W)  
    margins = data[0]["pred_boxes_XYXY"][0].to(torch.int) # tensor
    padding = [margins[0], img.size(1) - margins[3], img.size(2) - margins[2], margins[1]]
    iuv = pad(iuv, padding)
    assert iuv.size() == img.size() # (3,H,W)

    # Mask densepose image
    filename = INPUT_IMG.rsplit('.', 1)[0]
    mask = decode_image(os.path.join(HUMANMATTING['output'], filename + '.png')).to(torch.get_default_device()) # (1,H,W)

    # normalize the mask to 0,1
    mask[mask <= MASK_THRESHOLD] = 0
    mask[mask > MASK_THRESHOLD] = 1
    iuv *= mask

    # apply i==0 mask, note that unsqueeze+expand don't copy, repeat does copy
    iuv[(iuv[0,:,:] == 0).unsqueeze(0).expand([3, -1, -1])] = 0

    write_png(iuv.cpu(), os.path.join('data', 'densepose', 'densepose.png'))


def img2texture():
    '''
    Create masked densepose image from the input image,
    and then use them to construct partial Atlas&SMPL textures and masks.
    '''
    # Create masked densepose
    # img2densepose()

    img = decode_image(DENSEPOSE['input']).to(torch.get_default_device())
    iuv = decode_image(os.path.join('data', 'densepose', 'densepose.png')).to(torch.get_default_device())
    assert img.size() == iuv.size()
    _, h, w = img.size()

    # i==0 mask on image removes its background
    img[(iuv[0,:,:] == 0).unsqueeze(0).expand([3, -1, -1])] = 0

    # iuv dtype conversion
    iuv = torch.vstack([iuv[0].unsqueeze(0), iuv[1:3] / 255. * (ATLAS_PART_LEN - 1)]).int() # int32
    
    # Construct warp kernel for atlas texture and mask
    import warp as wp
    atlas_textures = wp.zeros((24, ATLAS_PART_LEN, ATLAS_PART_LEN), dtype=wp.vec3i)
    atlas_masks = wp.zeros((24, ATLAS_PART_LEN, ATLAS_PART_LEN), dtype=wp.int32)
    img = wp.from_torch(img.permute(1, 2, 0).int(), dtype=wp.vec3i) # (H,W)
    iuv = wp.from_torch(iuv, dtype=wp.int32) # (3,H,W)

    @wp.kernel
    def img2atlas(image: wp.array2d(dtype=wp.vec3i), # type: ignore
                  iuv_data: wp.array3d(dtype=wp.int32), # type: ignore
                  texture: wp.array3d(dtype=wp.vec3i), # type: ignore
                  mask: wp.array3d(dtype=wp.int32)): # type: ignore
        tid_h, tid_w = wp.tid()

        if iuv_data[0, tid_h, tid_w] != 0:
            texture[iuv_data[0, tid_h, tid_w] - 1,
                    iuv_data[1, tid_h, tid_w],
                    iuv_data[2, tid_h, tid_w]] = image[tid_h, tid_w]
            mask[iuv_data[0, tid_h, tid_w] - 1,
                 iuv_data[1, tid_h, tid_w],
                 iuv_data[2, tid_h, tid_w]] = 255
    
    wp.launch(kernel=img2atlas,
              dim=(h, w),
              inputs=[img, iuv, atlas_textures, atlas_masks])
    wp.synchronize()

    atlas_textures = wp.to_torch(atlas_textures).to(torch.uint8) # (N,H,W,C)
    atlas_texture = make_grid(atlas_textures.permute(0, 3, 1, 2), 6, 0) # (3,H,W)
    atlas_masks = wp.to_torch(atlas_masks).to(torch.uint8) # (N,H,W)
    atlas_mask = make_grid(atlas_masks.unsqueeze(1), 6, 0) # (3,H,W)

    # invert the mask for inpainting
    # atlas_mask = invert(atlas_mask)

    write_png(atlas_texture.cpu(), os.path.join('data', 'texture', 'atlas.png'))
    write_png(atlas_mask.cpu(), os.path.join('data', 'mask', 'atlas_mask.png'))

    # SMPL texture and mask
    import numpy as np
    from UVTextureConverter import Atlas2Normal
    converter = Atlas2Normal(atlas_size=ATLAS_PART_LEN, normal_size=SMPL_PART_LEN)
    smpl_texture, smpl_mask = converter.convert(atlas_textures.cpu().numpy(), atlas_masks.cpu().numpy())

    # the original project suggests to create the mask from smpl_texture, with <2 mask as 0
    # though correct, this smpl_mask cannot crop the edges of smpl texture,
    # and will affect inpainting results (inpainting usually comes with a padding)

    # the smpl_texture may be round() before casting to uint8
    # round() produce more accurate results
    # TODO check other possible round() castings
    smpl_texture = np.uint8((smpl_texture * 255).round()) # (L,L,3)

    # inverted padded mask
    smpl_inpaint_mask = (smpl_texture[:,:,0] < 2) & (smpl_texture[:,:,1] < 2) & (smpl_texture[:,:,2] < 2) # (L,L)

    smpl_inpaint_mask = np.uint8(smpl_inpaint_mask) * 255

    from PIL import Image
    Image.fromarray(smpl_texture).save(os.path.join('data', 'texture', 'smpl.png'))
    Image.fromarray(np.uint8(smpl_mask)).save(os.path.join('data', 'mask', 'smpl_mask.png'))
    Image.fromarray(smpl_inpaint_mask).save(os.path.join('data', 'mask', 'smpl_inpaint_mask.png'))


# DIFFUSERS = {
#     'model': 'stabilityai/stable-diffusion-2-base',
#     'input_dir': os.path.join('data', 'train'),
#     'class_dir': os.path.join('data', 'temp'),
#     'output_dir': os.path.join('data', 'trained_model', str(time.time_ns())),
#     'input_prompt': 'a smpl texturemap',
#     'class_prompt': 'a texturemap',
#     'resolution': SMPL_PART_LEN, # all training data will be resized into this resolution
#     'num_class_images': 10,
#     'max_train_steps': 500 * 3,
#     'checkpoint_steps': 500,
#     'gradient_accumulation_steps': 2, # larger means more stable update at cost of larger VRAM occupation
#     'lr_scheduler': 'constant', # cosine is better than constant
#     'learning_rate': 1e-6, # larger g_a_s above means bigger update, so smaller learning_rate is used
#     'prior_loss_weight': 1.0 # larger learns more conservatively, smaller more aggressively
# }

# def train():
#     '''
#     Finetune SD2 inpainting model (resolution 512) with DreamBooth.
#     '''
#     # reset class_dir (not necessary)
#     import subprocess
#     # subprocess.run(['rm', '-rf', DIFFUSERS['class_dir']])
#     # subprocess.run(['mkdir', '-p', DIFFUSERS['class_dir']])

#     cmd = ['accelerate', 'launch', os.path.join('diffusers', 'examples', 'dreambooth', 'train_dreambooth.py'), '--pretrained_model_name_or_path=' + DIFFUSERS['model'], '--instance_data_dir=' + DIFFUSERS['input_dir'], '--class_data_dir=' + DIFFUSERS['class_dir'], '--instance_prompt="' + DIFFUSERS['input_prompt'] + '"', '--class_prompt="' + DIFFUSERS['class_prompt'] + '"', '--with_prior_preservation', '--prior_loss_weight=' + str(DIFFUSERS['prior_loss_weight']), '--num_class_images=' + str(DIFFUSERS['num_class_images']), '--output_dir=' + DIFFUSERS['output_dir'], '--resolution=' + str(DIFFUSERS['resolution']), '--train_text_encoder', '--train_batch_size=1', '--max_train_steps=' + str(DIFFUSERS['max_train_steps']), '--checkpointing_steps=' + str(DIFFUSERS['checkpoint_steps']), '--gradient_accumulation_steps=' + str(DIFFUSERS['gradient_accumulation_steps']), '--gradient_checkpointing', '--learning_rate=' + str(DIFFUSERS['learning_rate']), '--use_8bit_adam', '--lr_scheduler=' + DIFFUSERS['lr_scheduler'], '--lr_warmup_steps=0', '--allow_tf32', '--mixed_precision=fp16']
    
#     subprocess.run(cmd)


# DIFFUSERS = {
#     'model': 'stabilityai/stable-diffusion-2-inpainting',
#     'input_dir': os.path.join('data', 'train'),
#     'class_dir': os.path.join('data', 'temp'),
#     'output_dir': os.path.join('data', 'trained_model', str(time.time_ns())),
#     'input_prompt': 'a smpl texturemap',
#     'class_prompt': 'a texturemap',
#     'resolution': SMPL_PART_LEN, # all training data will be resized into this resolution
#     'num_class_images': 10,
#     'max_train_steps': 500 * 3,
#     'checkpoint_steps': 500,
#     'gradient_accumulation_steps': 2, # larger means more stable update at cost of larger VRAM occupation
#     'lr_scheduler': 'constant', # cosine is better than constant
#     'learning_rate': 1e-6, # larger g_a_s above means bigger update, so smaller learning_rate is used
#     'prior_loss_weight': 1.0 # larger learns more conservatively, smaller more aggressively
# }

# def train():
#     '''
#     Finetune SD2 inpainting model (resolution 512) with DreamBooth.
#     '''
#     # reset class_dir (not necessary)
#     import subprocess
#     # subprocess.run(['rm', '-rf', DIFFUSERS['class_dir']])
#     # subprocess.run(['mkdir', '-p', DIFFUSERS['class_dir']])

#     cmd = ['accelerate', 'launch', os.path.join('diffusers', 'examples', 'research_projects', 'dreambooth_inpaint', 'train_dreambooth_inpaint.py'), '--pretrained_model_name_or_path=' + DIFFUSERS['model'], '--instance_data_dir=' + DIFFUSERS['input_dir'], '--class_data_dir=' + DIFFUSERS['class_dir'], '--instance_prompt="' + DIFFUSERS['input_prompt'] + '"', '--class_prompt="' + DIFFUSERS['class_prompt'] + '"', '--with_prior_preservation', '--prior_loss_weight=' + str(DIFFUSERS['prior_loss_weight']), '--num_class_images=' + str(DIFFUSERS['num_class_images']), '--output_dir=' + DIFFUSERS['output_dir'], '--resolution=' + str(DIFFUSERS['resolution']), '--train_text_encoder', '--train_batch_size=1', '--max_train_steps=' + str(DIFFUSERS['max_train_steps']), '--checkpointing_steps=' + str(DIFFUSERS['checkpoint_steps']), '--gradient_accumulation_steps=' + str(DIFFUSERS['gradient_accumulation_steps']), '--gradient_checkpointing', '--learning_rate=' + str(DIFFUSERS['learning_rate']), '--use_8bit_adam', '--lr_scheduler=' + DIFFUSERS['lr_scheduler'], '--lr_warmup_steps=0', '--mixed_precision=fp16']
    
#     subprocess.run(cmd)


def main():
    
    # img2texture()
    train()

if __name__ == '__main__':
    main()
