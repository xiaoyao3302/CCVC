import numpy as np
from PIL import Image


def create_pascal_label_colormap():
    """Creates a label colormap used in Pascal segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = 255 * np.ones((256, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]
    colormap[1] = [128, 0, 0]
    colormap[2] = [0, 128, 0]
    colormap[3] = [128, 128, 0]
    colormap[4] = [0, 0, 128]
    colormap[5] = [128, 0, 128]
    colormap[6] = [0, 128, 128]
    colormap[7] = [128, 128, 128]
    colormap[8] = [64, 0, 0]
    colormap[9] = [192, 0, 0]
    colormap[10] = [64, 128, 0]
    colormap[11] = [192, 128, 0]
    colormap[12] = [64, 0, 128]
    colormap[13] = [192, 0, 128]
    colormap[14] = [64, 128, 128]
    colormap[15] = [192, 128, 128]
    colormap[16] = [0, 64, 0]
    colormap[17] = [128, 64, 0]
    colormap[18] = [0, 192, 0]
    colormap[19] = [128, 192, 0]
    colormap[20] = [0, 64, 128]

    return colormap


def create_cityscapes_label_colormap():
    """Creates a label colormap used in CityScapes segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]

    return colormap


def colorize(mask, colormap):
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])
    for i in np.unique(mask):
        color_mask[mask == i] = colormap[i]

    return Image.fromarray(np.uint8(color_mask))