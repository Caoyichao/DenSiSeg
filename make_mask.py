import os
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import glob
import cv2

def main():
    mask_file_path = 'E:\\python\\Data\\MakeLable\\MakeLable\\MakeLable\\smoke_source'
    mask_files = sorted(glob.glob(mask_file_path + '/*.png'))
    mask_output = '../mask_output'
    show_output = '../show_output'
    radius = 20
    if not os.path.exists(mask_output):
        os.makedirs(mask_output)
    if not os.path.exists(show_output):
        os.makedirs(show_output)

    for file in mask_files:
        basename = os.path.basename(file).split('.')[0]
        mask = cv2.imread(file, 0)
        h, w = mask.shape
        # mask = mask / 255.
        indexes = np.argwhere(mask == 255)
        heatmaps = []
        try:
            for center_pos in indexes:
                heatmap = np.zeros((h, w), dtype=np.float)
                draw_umich_gaussian(heatmap, center_pos, radius)
                heatmaps.append(heatmap)
            heatmaps = np.array(heatmaps)
            heatmap = np.max(heatmaps, axis=0)
            #np.save(os.path.join(mask_output, basename + '.npy'), heatmap)
            
            show_map = heatmap * 255
            np.clip(show_map, 0, 255)
            show_map = show_map.astype('uint8')
            #cv2.imwrite(os.path.join(show_output, basename + '.png'), show_map)
        except:
            print(basename)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[1]), int(center[0])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

if __name__ == '__main__':
    main()