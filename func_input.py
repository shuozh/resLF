import imageio
import numpy as np
import skimage.color as color
import torch
"""
prepare the LF images for resLF testing
"""


def image_input(root, image_path, view_num_all):

    test_image = imageio.imread(root + image_path + '/lf.png')
    gt_image = imageio.imread(root + image_path + '/lf_h.png')

    # extract the y channel from ycbcr color space
    test_image_ycbcr = color.rgb2ycbcr(test_image)
    gt_image_ycbcr = color.rgb2ycbcr(gt_image)

    # change the angular resolution of LF images for different input
    if view_num_all < 9:
        test_image_input = angular_resolution_changes(test_image_ycbcr, 9, view_num_all)
        gt_image_input = angular_resolution_changes(gt_image_ycbcr, 9, view_num_all)
    else:
        test_image_input = test_image_ycbcr
        gt_image_input = gt_image_ycbcr

    return test_image_input[:, :, 0]/255.0, gt_image_input[:, :, 0]/255.0


def image_input_color(root, image_path):

    gt_image = imageio.imread(root + image_path + '/lf_h.png')
    angular_position = np.int8((9 - 1) / 2)
    # extract the y channel from ycbcr color space
    gt_image_ycbcr = color.rgb2ycbcr(gt_image[angular_position::9, angular_position::9, :])

    return gt_image_ycbcr


def angular_resolution_changes(image, view_num_ori, view_num_new):

    num_vew_gap = np.int8((view_num_ori - view_num_new)/2)

    image_h = image.shape[0] / view_num_ori
    image_w = image.shape[1] / view_num_ori
    nD = image.shape[2]

    new_image = np.zeros((int(image_h * view_num_new), int(image_w * view_num_new), nD), dtype=np.float32)

    for i in range(0, view_num_new, 1):
        for j in range(0, view_num_new, 1):
            new_image[i::view_num_new, j::view_num_new, :] = \
                image[i+num_vew_gap::view_num_ori, j+num_vew_gap::view_num_ori, :]

    return new_image


def multi_input_all(test_image, gt_image, view_num_ori, u, v):

    # angular resolution of the divided LF part
    mid_view_min = np.int8(max(min(u, view_num_ori-u-1, v, view_num_ori-v-1), 1))
    view_num_min = mid_view_min*2 + 1

    image_h = test_image.shape[0] / view_num_ori
    image_w = test_image.shape[1] / view_num_ori

    # extract the ground truth image view
    gt_data_tmp = torch.zeros((1, int(image_h * 2), int(image_w * 2)), dtype=torch.float32)
    gt_view = gt_image[u::view_num_ori, v::view_num_ori]
    gt_data_tmp[0, :, :] = gt_view[:, :]

    # initialize the input image stacks
    train_data_0 = torch.zeros((1, view_num_min, int(image_h), int(image_w)), dtype=torch.float32)
    train_data_90 = torch.zeros((1, view_num_min, int(image_h), int(image_w)), dtype=torch.float32)
    train_data_45 = torch.zeros((1, view_num_min, int(image_h), int(image_w)), dtype=torch.float32)
    train_data_135 = torch.zeros((1, view_num_min, int(image_h), int(image_w)), dtype=torch.float32)

    for i in range(0, view_num_min, 1):

        if ((v - mid_view_min + i) >= 0) and ((v - mid_view_min + i) < view_num_ori):
            img_tmp = test_image[u::view_num_ori, (v-mid_view_min+i)::view_num_ori]
            train_data_0[0, i, :, :] = img_tmp
            if ((u - mid_view_min + i) >= 0) and ((u - mid_view_min + i) < view_num_ori):
                img_tmp = test_image[(u-mid_view_min+i)::view_num_ori, (v-mid_view_min+i)::view_num_ori]
                train_data_45[0,i, :, :] = img_tmp

        if ((u - mid_view_min + i) >= 0) and ((u - mid_view_min + i) < view_num_ori):
            img_tmp = test_image[u - mid_view_min + i::view_num_ori, v::view_num_ori]
            train_data_90[0,i, :, :] = img_tmp
            if ((v + mid_view_min - i) >= 0) and ((v + mid_view_min - i) < view_num_ori):
                img_tmp = test_image[u - mid_view_min + i::view_num_ori, (v + mid_view_min - i)::view_num_ori]
                train_data_135[0,i, :, :] = img_tmp

    return train_data_0, train_data_90, train_data_45, train_data_135, gt_data_tmp
