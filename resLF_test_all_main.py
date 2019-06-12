import torch
import os
import scipy.io
import numpy as np
from skimage.measure import compare_ssim
from math import log10

from resLF_model import resLF
from func_input import multi_input_all, image_input

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def main():

    # parameters setting
    dir_images = 'hci_testing/'
    dir_save_path = 'output_images/'

    # we currently support LFSR with 7*7 angular resolution
    view_num_all = 7
    scale = 2
    save_flag = 1

    for root, dirs, files in os.walk(dir_images):

        if len(dirs) == 0:
            break

        for image_path in dirs:
            test_image, gt_image = image_input(root, image_path, view_num_all)

            test_image, gt_image = \
                torch.from_numpy(test_image.copy()), torch.from_numpy(gt_image.copy())
            torch.no_grad()

            psnr_image = np.zeros((view_num_all, view_num_all))
            ssim_image = np.zeros((view_num_all, view_num_all))
            image_h = gt_image.shape[0] / view_num_all
            image_w = gt_image.shape[1] / view_num_all
            pre_lf = np.zeros((int(view_num_all), int(view_num_all), int(image_h), int(image_w)), dtype=np.float32)

            # model reading
            model7, model5, model3, model3v, model3h, model3hv = resLF_model_reading(scale)

            # for central image
            u_list = [3]
            v_list = [3]
            psnr_image, ssim_image, pre_lf = test_all(test_image, gt_image, view_num_all, u_list, v_list, model7,
                                               psnr_image, ssim_image, pre_lf)

            # for images in 3*3 border
            u_list = [2, 3, 4, 2, 4, 2, 3, 4]
            v_list = [2, 2, 2, 3, 3, 4, 4, 4]
            psnr_image, ssim_image, pre_lf = test_all(test_image, gt_image, view_num_all, u_list, v_list, model5,
                                               psnr_image, ssim_image, pre_lf)

            # for image sin 5*5 border
            u_list = [1, 2, 3, 4, 5, 1, 5, 1, 5, 1, 5, 1, 2, 3, 4, 5]
            v_list = [1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5]
            psnr_image, ssim_image, pre_lf = test_all(test_image, gt_image, view_num_all, u_list, v_list, model3,
                                               psnr_image, ssim_image, pre_lf)

            # for left and right border images
            u_list = [0, 0, 0, 0, 0, 6, 6, 6, 6, 6]
            v_list = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
            psnr_image, ssim_image, pre_lf = test_all(test_image, gt_image, view_num_all, u_list, v_list, model3v,
                                               psnr_image, ssim_image, pre_lf)

            # for up and down image border images
            u_list = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
            v_list = [0, 0, 0, 0, 0, 6, 6, 6, 6, 6]
            psnr_image, ssim_image, pre_lf = test_all(test_image, gt_image, view_num_all, u_list, v_list, model3h,
                                               psnr_image, ssim_image, pre_lf)

            # for corner images
            u_list = [0, 6, 0, 6]
            v_list = [0, 6, 6, 0]
            psnr_image, ssim_image, pre_lf = test_all(test_image, gt_image, view_num_all, u_list, v_list, model3hv,
                                               psnr_image, ssim_image, pre_lf)

            print('=>{} SR PSNR Avr: {:.4f}, Max: {:.4f}, Min: {:.4f}, SSIM: Avr: {:.4f}, Max: {:.4f}, Min: {:.4f}'
                  .format(image_path, np.mean(psnr_image), np.max(psnr_image), np.min(psnr_image),
                          np.mean(ssim_image), np.max(ssim_image), np.min(ssim_image)))

            if save_flag:
                scipy.io.savemat(dir_save_path + image_path + '_result.mat', {'pre_lf': pre_lf})  # 写入mat文件


def resLF_model_reading(scale):

    model7 = resLF(n_view=7, scale=scale)
    model7.cuda()
    state_dict = torch.load('model_all/' + str(7) + '.pkl')
    model7.load_state_dict(state_dict)

    model5 = resLF(n_view=5, scale=scale)
    model5.cuda()
    state_dict = torch.load('model_all/' + str(5) + '.pkl')
    model5.load_state_dict(state_dict)

    model3 = resLF(n_view=3, scale=scale)
    model3.cuda()
    state_dict = torch.load('model_all/' + str(3) + '.pkl')
    model3.load_state_dict(state_dict)

    model3v = resLF(n_view=3, scale=scale)
    model3v.cuda()
    state_dict = torch.load('model_all/' + str(3) + '_v.pkl')
    model3v.load_state_dict(state_dict)

    model3h = resLF(n_view=3, scale=scale)
    model3h.cuda()
    state_dict = torch.load('model_all/' + str(3) + '_h.pkl')
    model3h.load_state_dict(state_dict)

    model3hv = resLF(n_view=3, scale=scale)
    model3hv.cuda()
    state_dict = torch.load('model_all/' + str(3) + '_hv.pkl')
    model3hv.load_state_dict(state_dict)

    return model7, model5, model3, model3v, model3h, model3hv


def test_all(test_image, gt_image, view_num_all, u_list, v_list, model, psnr_image, ssim_image, pre_lf):

    for i in range(0, len(u_list), 1):
        u = u_list[i]
        v = v_list[i]

        model.eval()
        train_data_0, train_data_90, train_data_45, train_data_135, gt_data = \
            multi_input_all(test_image, gt_image, view_num_all, u, v)

        train_data_0, train_data_90, train_data_45, train_data_135, gt_data = \
            train_data_0.cuda(), train_data_90.cuda(), train_data_45.cuda(), train_data_135.cuda(), gt_data.cuda()

        # Forward pass: Compute predicted y by passing x to the model
        with torch.no_grad():
            gt_pred = model(train_data_0, train_data_90, train_data_45, train_data_135)

        # calculate the PSNR and SSIM values
        output = gt_pred[0, 0, :, :]
        img_pre = output.cpu().numpy()
        img_pre = np.clip(img_pre, 0.0, 1.0)

        output = gt_data[0, :, :]
        gt_img = output.cpu().numpy()
        image_h = gt_img.shape[0]
        image_w = gt_img.shape[1]

        compare_loss = (img_pre - gt_img) ** 2
        compare_loss = compare_loss.sum() / (int(image_w) * int(image_h))
        psnr = 10 * log10(1 / compare_loss)
        ssim = compare_ssim(img_pre, gt_img)

        psnr_image[u, v] = psnr
        ssim_image[u, v] = ssim
        pre_lf[u, v, :, :] = img_pre

    return psnr_image, ssim_image, pre_lf


if __name__ == '__main__':
    main()