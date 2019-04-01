INSTANCE_ID = '9d336a61-392e-49f5-8f03-52023c515d7e'
from sentinelhub import WmsRequest, WcsRequest, MimeType, CRS, BBox


import datetime
import numpy as np

import matplotlib.pyplot as plt

def plot_image(image, factor=1):
    """
    Utility function for plotting RGB images.
    """
    fig = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))

    if np.issubdtype(image.dtype, np.floating):
        plt.imshow(np.minimum(image * factor, 1))
    else:
        plt.imshow(image)

betsiboka_coords_wgs84 = [46.16, -16.15, 46.51, -15.58]
betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
wms_true_color_request = WmsRequest(layer='TRUE-COLOR-S2-L1C',
                                    bbox=betsiboka_bbox,
                                    time='2017-12-15',
                                    width=512, height=856,
                                    instance_id=INSTANCE_ID)
wms_true_color_img = wms_true_color_request.get_data()
print('Returned data is of type = %s and length %d.' % (type(wms_true_color_img), len(wms_true_color_img)))
plot_image(wms_true_color_img[-1])
plt.show()