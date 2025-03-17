from torchvision import transforms


def equalize(img):
    return transforms.functional.equalize(img)
