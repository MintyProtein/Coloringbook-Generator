import torch
import torch.nn as nn
from . import models

# returns a pretrained vgg
def build_vgg(checkpoint_path, device=torch.device('cuda')):
    vgg = models.vgg
    if checkpoint_path is not None:
        vgg.load_state_dict(torch.load(checkpoint_path))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    vgg.to(device)
    return vgg

# returns a pretrained decoder
def build_decoder(checkpoint_path, device=torch.device('cuda')):
    decoder = models.decoder
    if checkpoint_path is not None:
        decoder.load_state_dict(torch.load(checkpoint_path))
    decoder.to(device)
    return decoder

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

# style transfer the content image with style image
def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None, device=torch.device('cuda')):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    # adjust the degree of stylization by alpha
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)
