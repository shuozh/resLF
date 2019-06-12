import torch
import os
import numpy as np
from skimage.measure import compare_ssim
from math import log10
from PIL import Image
from resLF_model import resLF
from func_input import multi_input_all, image_input, image_input_color
import skimage.color as color

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def main():

    # We currently provide the central view SR model with 3*3\5*5\7*7\9*9 LF input for *2 magnification scale
    view_n = 9  # angular resolution (3,5,7,9)
    scale = 2

    # parameters setting
    dir_images = 'testing_data/'
    dir_save_path = 'output_images/'
    save_flag = 1

    model = resLF(n_view=view_n, scale=scale)
    model.cuda()
    state_dict = torch.load('model_all/' + str(view_n) + '.pkl')  # loading the SR model
    model.load_state_dict(state_dict)

    for root, dirs, files in os.walk(dir_images):

        if len(dirs) == 0:
            break
        for image_path in dirs:

            test_image, gt_image = image_input(root, image_path, view_n)
            test_image, gt_image = torch.from_numpy(test_image.copy()), torch.from_numpy(gt_image.copy())
            torch.no_grad()

            # for central image
            angular_position = np.int8((view_n-1)/2)
            psnr_image, ssim_image, img_pre = test_all(test_image, gt_image, view_n, angular_position, model)

            print('=>{} SR PSNR: {:.4f}, SSIM: {:.4f}'.format(image_path, psnr_image, ssim_image))

            if save_flag:
                
                img_pre = np.clip(img_pre*255.0, 16.0, 235.0)
                img = Image.fromarray(np.uint8(img_pre))
                img.save(dir_save_path + image_path + '_img_pre.png')

                # calculate the super-resolved color images
                img_ycbcr = image_input_color(root, image_path)
                img_ycbcr[:, :, 0] = img_pre
                img_rgb = color.ycbcr2rgb(img_ycbcr)
                img_rgb = np.clip(img_rgb,0.0,1.0)
                img = Image.fromarray(np.uint8(img_rgb*255.0))
                img.save(dir_save_path + image_path + '_pre_color.png')


def test_all(test_image, gt_image, view_n, angular_position, model):

    model.eval()
    train_data_0, train_data_90, train_data_45, train_data_135, gt_data = \
        multi_input_all(test_image, gt_image, view_n, angular_position, angular_position)

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

    return psnr, ssim, img_pre


if __name__ == '__main__':
    main()
