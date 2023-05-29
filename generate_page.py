import cv2
import torch
from pytorch_AdaIN import build_decoder, build_vgg, style_transfer
from utils import postprocess
from utils.data import get_transforms

""" 
Functions to transfer the input image to coloring page with each methods.

To get a desired output, the input image should be preprocessed by SAM and utils functions

"""

# edge detection
def generate_page_canny(content):
    gen_img = cv2.Canny(content, 60, 150)
    gen_img = cv2.bitwise_not(gen_img)
    return gen_img

# AdaIN
def generate_page_adain(content, vgg_path='./model_checkpoints/vgg.pth', decoder_path='./model_checkpoints/decoder.pth', style_path='./assets/style.jpg', device='cuda'):
    with torch.no_grad():
        H, W, C = content.shape
        # use prepared image from style_path for style image
        style = cv2.imread(style_path)
        style = cv2.resize(style, (H, W))
        
        vgg = build_vgg(checkpoint_path=vgg_path, device=device)
        decoder = build_decoder(checkpoint_path=decoder_path, device=device)
        transforms = get_transforms()

        content = transforms(content)[None, :]
        content = content.to(device)
        style = transforms(style)[None, :]
        style = style.to(device)
        
        # AdaIN transfer
        gen_img = style_transfer(vgg, decoder, content, style)[0]
        # convert torch.Tensor to nunmpy image
        gen_img = gen_img.cpu().numpy().transpose(1,2,0)
        gen_img = postprocess(gen_img)
        return gen_img