from torchvision.transforms import Compose, ToTensor, Normalize

# returns a torchvision transforms object
def get_transforms(normalize=True):
    transforms_list = [ToTensor()]
    if normalize:
        transforms_list.append(Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    return Compose(transforms_list)