import datetime
import matplotlib.pyplot as plt
import numpy as np

from sentinelhub import WmsRequest, BBox, CRS, MimeType, CustomUrlParam, get_area_dates
from s2cloudless import S2PixelCloudDetector, CloudMaskRequest

def overlay_cloud_mask(image, mask=None, factor=1./255, figsize=(15, 15), fig=None):
    """
    Utility function for plotting RGB images with binary mask overlayed.
    """
    if fig == None:
        plt.figure(figsize=figsize)
    rgb = np.array(image)
    plt.imshow(rgb * factor)
    if mask is not None:
        cloud_image = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        cloud_image[mask == 1] = np.asarray([255, 255, 0, 100], dtype=np.uint8)
        plt.imshow(cloud_image)


def plot_probability_map(rgb_image, prob_map, factor=1./255, figsize=(15, 30)):
    """
    Utility function for plotting a RGB image and its cloud probability map next to each other.
    """
    plt.figure(figsize=figsize)
    plot = plt.subplot(1, 2, 1)
    plt.imshow(rgb_image * factor)
    plot = plt.subplot(1, 2, 2)
    plot.imshow(prob_map, cmap=plt.cm.inferno)


def plot_cloud_mask(mask, figsize=(15, 15), fig=None):
    """
    Utility function for plotting a binary cloud mask.
    """
    if fig == None:
        plt.figure(figsize=figsize)
    plt.imshow(mask, cmap=plt.cm.gray)




INSTANCE_ID = '9d336a61-392e-49f5-8f03-52023c515d7e'
LAYER_NAME = 'TRUE-COLOR-S2-L1C'  # e.g. TRUE-COLOR-S2-L1C
bbox_coords_wgs84 = [8.279572, 46.560749, 8.463593, 46.672528]
bounding_box = BBox(bbox_coords_wgs84, crs=CRS.WGS84)
wms_true_color_request = WmsRequest(layer=LAYER_NAME,
                                    bbox=bounding_box,
                                    time=('2018-09-23', '2018-09-25'),
                                    width=600, height=None,
                                    image_format=MimeType.PNG,
                                    instance_id=INSTANCE_ID)
wms_true_color_imgs = wms_true_color_request.get_data()
bands_script = 'return [B01,B02,B04,B05,B08,B8A,B09,B10,B11,B12]'
wms_bands_request = WmsRequest(layer=LAYER_NAME,
                               custom_url_params={
                                   CustomUrlParam.EVALSCRIPT: bands_script,
                                   CustomUrlParam.ATMFILTER: 'NONE'
                               },
                               bbox=bounding_box,
                               time=('2018-09-23', '2018-09-25'),
                               width=600, height=None,
                               image_format=MimeType.TIFF_d32f,
                               instance_id=INSTANCE_ID)

wms_bands = wms_bands_request.get_data()
cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2)
cloud_probs = cloud_detector.get_cloud_probability_maps(np.array(wms_bands))
image_idx = 0
plot_probability_map(wms_true_color_imgs[image_idx], cloud_probs[image_idx])
plt.show()



