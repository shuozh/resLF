import imageio
import numpy as np
import skimage.color as color
import torch
import cv2
from PIL import Image
import os

"""
prepare the LF images for resLF testing
"""


def image_input(image_path, scale, view_num_all, view_num_ori, interpolation):
    """
    Prepare ground truth (high-resolution) light field and the related low-resolution light field 
    :param image_path: loading 4D training LF from this path
    :param scale: spatial upsampling scale
    :param view_num_all: crop Length from Initial LF for test
    :param view_num_ori: original angular resolution
    :param interpolation: downsampling interpolation methods. (blur or bicubic)
    :return: ground truth and test data with YCbCr
    """
    gt_image = imageio.imread(image_path)
    gt_image_ycbcr = color.rgb2ycbcr(gt_image[:, :, :3])

    # change the angular resolution of LF images for different input
    num_vew_gap = (view_num_ori + 1 - view_num_all) // 2

    image_h = gt_image_ycbcr.shape[0] // view_num_ori
    image_w = gt_image_ycbcr.shape[1] // view_num_ori
    channel_n = gt_image_ycbcr.shape[2]

    # cut the extra pixels
    if image_h % scale != 0:
        gt_image_ycbcr = gt_image_ycbcr[:-(image_h % scale) * view_num_ori, :, :]
        image_h -= image_h % scale
    if image_w % scale != 0:
        gt_image_ycbcr = gt_image_ycbcr[:, :-(image_w % scale) * view_num_ori, :]
        image_w -= image_w % scale

    # downsampling with interpolation
    gt_ycbcr = np.zeros((image_h * view_num_all, image_w * view_num_all, channel_n), dtype=np.float32)
    lr_ycbcr = np.zeros((image_h * view_num_all // scale, image_w * view_num_all // scale, channel_n), dtype=np.float32)

    for i in range(0, view_num_all, 1):
        for j in range(0, view_num_all, 1):
            gt_ycbcr[i::view_num_all, j::view_num_all, :] = gt_image_ycbcr[i + num_vew_gap::view_num_ori,
                                                            j + num_vew_gap::view_num_ori, :]
            if interpolation == 'blur':
                # interpolation with blur
                gt_ycbcr_blur = cv2.blur(
                    gt_image_ycbcr[i + num_vew_gap::view_num_ori, j + num_vew_gap::view_num_ori, :],
                    (scale, scale))
                lr_ycbcr[i::view_num_all, j::view_num_all, :] = gt_ycbcr_blur[scale // 2::scale, scale // 2::scale]
            else:
                # interpolation with bicubic
                for k in range(3):
                    img = Image.fromarray(
                        np.float32(gt_image_ycbcr[i + num_vew_gap::view_num_ori, j + num_vew_gap::view_num_ori, k]))
                    lr_ycbcr[i::view_num_all, j::view_num_all, k] = img.resize(
                        (image_w // scale, image_h // scale), Image.BICUBIC)

    return gt_ycbcr, lr_ycbcr


def multi_input_all(test_image, gt_image, view_num_ori, u, v, scale):
    """
    Find the maximum avaliale views centered at the reference view and the around avaliale 'star-like' views are stacked into 4 groups. 
    :param test_image: low-resolution light field
    :param gt_image: high-resolution light field
    :param view_num_ori: view number of light field
    :param u,v: angular coordinate of the reference view
    :param scale: downsampling scale
    :return: 4 image stacks and the high-resolution center view 
    """
    # angular resolution of the divided LF part
    mid_view_min = np.int8(max(min(u, view_num_ori - u - 1, v, view_num_ori - v - 1), 1))
    view_num_min = mid_view_min * 2 + 1

    image_h = test_image.shape[0] // view_num_ori
    image_w = test_image.shape[1] // view_num_ori

    # extract the ground truth image view
    gt_data_tmp = torch.zeros((1, image_h * scale, image_w * scale), dtype=torch.float32)
    gt_view = gt_image[u::view_num_ori, v::view_num_ori]
    gt_data_tmp[0, :, :] = gt_view[:, :]

    # initialize the input image stacks
    train_data_0 = torch.zeros((1, view_num_min, image_h, image_w), dtype=torch.float32)
    train_data_90 = torch.zeros((1, view_num_min, image_h, image_w), dtype=torch.float32)
    train_data_45 = torch.zeros((1, view_num_min, image_h, image_w), dtype=torch.float32)
    train_data_135 = torch.zeros((1, view_num_min, image_h, image_w), dtype=torch.float32)

    for i in range(0, view_num_min, 1):

        if ((v - mid_view_min + i) >= 0) and ((v - mid_view_min + i) < view_num_ori):
            img_tmp = test_image[u::view_num_ori, (v - mid_view_min + i)::view_num_ori]
            train_data_0[0, i, :, :] = img_tmp
            if ((u - mid_view_min + i) >= 0) and ((u - mid_view_min + i) < view_num_ori):
                img_tmp = test_image[(u - mid_view_min + i)::view_num_ori, (v - mid_view_min + i)::view_num_ori]
                train_data_45[0, i, :, :] = img_tmp

        if ((u - mid_view_min + i) >= 0) and ((u - mid_view_min + i) < view_num_ori):
            img_tmp = test_image[u - mid_view_min + i::view_num_ori, v::view_num_ori]
            train_data_90[0, i, :, :] = img_tmp
            if ((v + mid_view_min - i) >= 0) and ((v + mid_view_min - i) < view_num_ori):
                img_tmp = test_image[u - mid_view_min + i::view_num_ori, (v + mid_view_min - i)::view_num_ori]
                train_data_135[0, i, :, :] = img_tmp

    return train_data_0, train_data_90, train_data_45, train_data_135, gt_data_tmp


def uv_list_by_n(view_n):
    """
    caculate view index of each task
    :param view_n:
    :return: dictionary of tasks
    """
    range_list = []
    for item in range(view_n // 2):
        range_list.append([(item ** 2) * 2, view_n - 2 * item])
    uv_dic = {}
    for item in range(3, view_n + 1, 2):
        if item == 3:  # in the corner or border
            uv_dic['u3h'] = []
            uv_dic['v3h'] = []
            uv_dic['u3v'] = []
            uv_dic['v3v'] = []
            uv_dic['u3hv'] = []
            uv_dic['v3hv'] = []

        uv_dic['u' + str(item)] = []
        uv_dic['v' + str(item)] = []

    """
    distance matrix ( 7 * 7 )
    each task has different range of distance to central
    distance = (view_n_central - i) ** 2 + (view_n_central - j) ** 2
    18	13	10	9	10	13	18
    13	8	5	4	5	8	13
    10	5	2	1	2	5	10
    9	4	1	0	1	4	9 
    10	5	2	1	2	5	10
    13	8	5	4	5	8	13
    18	13	10	9	10	13	18
    """
    view_n_central = view_n // 2
    for i in range(view_n):
        for j in range(view_n):
            if i in [0, view_n - 1] and j in [0, view_n - 1]:
                uv_dic['u3hv'].append(i)
                uv_dic['v3hv'].append(j)
            elif i not in [0, view_n - 1] and j in [0, view_n - 1]:
                uv_dic['u3h'].append(i)
                uv_dic['v3h'].append(j)
            elif i in [0, view_n - 1] and j not in [0, view_n - 1]:
                uv_dic['u3v'].append(i)
                uv_dic['v3v'].append(j)
            else:
                distance = (view_n_central - i) ** 2 + (view_n_central - j) ** 2
                for item in range_list:
                    if distance <= item[0]:
                        uv_dic['u' + str(item[1])].append(i)
                        uv_dic['v' + str(item[1])].append(j)
                        break
    return uv_dic
