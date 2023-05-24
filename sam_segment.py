from segment_anything import sam_model_registry, SamPredictor

# returns pretrained SamPredictor
def build_sam(model_type="vit_h",
              checkpoint_path="./model_checkpoints/sam_vit_h_4b8939.pth",
              device='cuda'):
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    return predictor

# returns the predicted mask from given bbox
def predict_mask_with_box(predictor, image, box):
    predictor.set_image(image)
    
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],
        multimask_output=False,
    )
    
    return masks