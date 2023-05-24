import torch
import gradio as gr
from utils import *
from utils.gradio import *
from adain_transfer import *
from sam_segment import build_sam, predict_mask_with_box

device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
    

with gr.Blocks() as demo:
    # SamPredictor, wrapped by gradio.State
    sam = gr.State(build_sam())
    
    # coordinates of bbox: [[x1, y1], [x2, y2]]
    bbox_coords = gr.State([[None, None], [None, None]])
    
    with gr.Row():
        # Col A: The input image
        with gr.Column(scale=1):
            input_img = gr.Image(label='Input image', show_label=False).style(height=500)
            reset_btn = gr.Button(value='Reset')
        
        # Col B: The bbox + segmented image
        with gr.Column(scale=1):
            segment_img = gr.Image(label='Segment image', interactive=False).style(height=500)
            ratio_slider = gr.Slider(label='Background ratio', minimum=1, maximum=3, value=1.3, step=0.1)
            segment_btn = gr.Button(value='Segment')
        
        #Col C: The final image
        with gr.Column(scale=1):
            output_img = gr.Image(label='Generated image', interactive=False).style(height=500)
            method_dropdown = gr.Dropdown(["AdaIN", "Edge Detection"], value="AdaIN", label="Transfer Method")
            generate_btn = gr.Button(value='Generate')
    
    # set bbox coordinates by clicking the input image
    def on_image_selected(bbox, input_img, evt:gr.SelectData):
        
        # if neither of the 2 points is selected yet
        if bbox[0] == [None, None]:
            bbox[0] = evt.index
            new_img = add_point(input_img, bbox[0]) 
        # if the first point is selected
        elif bbox[1] == [None, None]:
            bbox[1] = evt.index
            new_img = add_bbox(input_img, bbox)
        # if all points are already selected -- nothing changes
        else:
            new_img = add_bbox(input_img, bbox)
            
        return {bbox_coords: bbox,
                segment_img: new_img}
    
    input_img.select(fn=on_image_selected,
                     inputs=[bbox_coords, input_img],
                     outputs=[bbox_coords, segment_img])
    
    # reset components
    def reset_components():
        return {bbox_coords: [[None, None], [None, None]],
                segment_img: None,
                output_img: None}
    input_img.change(fn=reset_components,
                     outputs=[bbox_coords, segment_img, output_img])
    reset_btn.click(fn=reset_components,
                    outputs=[bbox_coords, segment_img, output_img])
    
    # segment the image, and resize it by padding
    def on_segment_clicked(bbox, input_img, ratio_slider, sam):
        mask = predict_mask_with_box(sam, input_img, np.array(bbox))[0]
        segment_img =  remove_background(input_img, mask)
        segment_img = center_object(segment_img, mask, ratio=ratio_slider)
        return segment_img
    segment_btn.click(fn=on_segment_clicked,
                      inputs=[bbox_coords, input_img, ratio_slider, sam],
                      outputs=segment_img)
    
    # generate the final image by choosen method
    def on_generate_clicked(segment_img, method_dropdown):
        gen_img = None
        # read the choosen method from method_dropdown
        if method_dropdown == "AdaIN":
            gen_img = generate_page_adain(segment_img)
        elif method_dropdown == "Edge Detection":
            gen_img = generate_page_canny(segment_img)
        return gen_img
    generate_btn.click(fn=on_generate_clicked,
                       inputs=[segment_img, method_dropdown],
                       outputs=output_img)

demo.launch(share=True)