# ColoringBook Generator
- Convert your images to coloring pages

## Installation
## Usage
Download the model checkpoints, and put them into ./model_checkpoints
- [AdaIN](https://drive.google.com/drive/folders/1GEb1KGGMdy02wDxu85_IIgNv5cXyQTex): vgg.pth, decoder.pth
- [Segment Anything](https://github.com/facebookresearch/segment-anything#model-checkpoints): [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

## Demo
- Demo UI built with gradio
```
python app_gradio.py
```