import random
import re
import sys

import cv2
import numpy as np
from PIL import Image
import pylab

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


class RandomCrop(object):
    def __init__(self, CropSize=0.1):
        self.CropSize = CropSize

    def __call__(self, image, normal):
        h, w = normal.shape[:2]
        img_h, img_w = image.shape[:2]
        CropSize_w, CropSize_h = max(1, int(w * self.CropSize)), max(1, int(h * self.CropSize))
        x1, y1 = random.randint(0, CropSize_w), random.randint(0, CropSize_h)
        x2, y2 = random.randint(w - CropSize_w, w), random.randint(h - CropSize_h, h)

        normal_crop = normal[y1:y2, x1:x2]
        normal_resize = cv2.resize(normal_crop, (w, h), interpolation=cv2.INTER_NEAREST)

        image_crop = image[4 * y1:4 * y2, 4 * x1:4 * x2]
        image_resize = cv2.resize(image_crop, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        # import matplotlib.pyplot as plt
        # plt.subplot(2, 3, 1)
        # plt.imshow(image)
        # plt.subplot(2, 3, 2)
        # plt.imshow(image_crop)
        # plt.subplot(2, 3, 3)
        # plt.imshow(image_resize)
        #
        # plt.subplot(2, 3, 4)
        # plt.imshow((normal + 1.0) / 2, cmap="rainbow")
        # plt.subplot(2, 3, 5)
        # plt.imshow((normal_crop + 1.0) / 2, cmap="rainbow")
        # plt.subplot(2, 3, 6)
        # plt.imshow((normal_resize + 1.0) / 2, cmap="rainbow")
        # plt.show()
        # plt.pause(1)
        # plt.close()

        return image_resize, normal_resize


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]

    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))

    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data


def write_cam(file, cam):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')
    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) +
            ' '+ '0.0'+ '\n')
    f.close()

def write_cam2(path, list):
    f = open(path, "w")
    for i in range(len(list)):
        f.write(str(list[i]))
        f.write('\n')
    f.close()
