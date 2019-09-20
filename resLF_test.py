import torch
import os
import numpy as np
from skimage.measure import compare_ssim
from math import log10
import cv2
import skimage.color as color

from resLF_model import resLF
from func_input import multi_input_all, image_input, uv_list_by_n
import sys
import time
import pandas as pd
from argparse import ArgumentParser, ArgumentTypeError


class Logger:
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Unsupported value encountered.')


def opts_parser():
    usage = "resLF Test"
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-I', '--image_path', type=str, default=None, dest='image_path',
        help='Loading 4D LF images from this path: (default: %(default)s)')
    parser.add_argument(
        '-M', '--model', type=str, default='model_all/', dest='model_path',
        help='Loading pre-trained model file from this path: (default: %(default)s)')
    parser.add_argument(
        '-S', '--save_path', type=str, default='test_result/', dest='save_path',
        help='Save upsampled LF to this path: (default: %(default)s)')
    parser.add_argument(
        '-o', '--original_length', type=int, default=14, dest='original_length',
        help='Original light field angular resolution: (default: %(default)s)')
    parser.add_argument(
        '-c', '--crop_length', type=int, default=7, dest='crop_length',
        help='Crop light field with different angular resolution for test: (default: %(default)s)')
    parser.add_argument(
        '-s', '--scale', type=int, default=2, dest='scale',
        help='Spatial upsampling scale: (default: %(default)s)')
    parser.add_argument(
        '-C', '--central', type=str2bool, default=True, dest='is_single',
        help='Only super-resolve central view: (default: %(default)s)')
    parser.add_argument(
        '-i', '--interpolation', type=str, default='blur', dest='interpolation',
        help='downsampling interpolation method (`blur`, `bicubic`): (default: %(default)s)')
    parser.add_argument(
        '-g', '--gpu_no', type=int, default=0, dest='gpu_no',
        help='GPU used: (default: %(default)s)')

    return parser


def main(image_path, model_path, save_path='result/', view_n_ori=14, view_n=7, scale=2, is_single=True,
         interpolation='blur', gpu_no=0):
    inter_type = ('bicubic', 'blur')
    if interpolation not in inter_type:
        raise ValueError('`{}` interpolation is not supported, Possible values are: bicubic, blur'.format(interpolation))

    # choose GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)

    print('=' * 40)
    print('create save directory...')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sys.stdout = Logger(save_path + 'test_{}.log'.format(int(time.time())), sys.stdout)
    print('done')
    print('=' * 40)
    print('build network and load model...')
    if is_single:
        model = resLF_model_reading_single(model_path, view_n, scale, interpolation)
    else:
        model = resLF_model_reading_all(model_path, view_n, scale, interpolation)
    print('done')
    print('=' * 40)
    print('predict image...')

    xls_list = []
    psnr_list = []
    ssim_list = []
    time_list = []

    files = os.listdir(image_path)
    for index, image_name in enumerate(files):
        print('-' * 100)
        print('[{}/{}]'.format(index + 1, len(files)), image_name)
        time_item_start = time.time()
        gt_ycbcr, lr_ycbcr = image_input(image_path + image_name, scale, view_n, view_n_ori, interpolation)
        lr_y, gt_hr_y, lr_cbcr = lr_ycbcr[:, :, 0] / 255.0, gt_ycbcr[:, :, 0] / 255.0, lr_ycbcr[:, :, 1:3]

        psnr_image, ssim_image, hr_y = predict_y(lr_y, gt_hr_y, view_n, is_single, model, scale)

        hr_cbcr = predict_cbcr(lr_cbcr, scale, view_n, is_single)
        time_ = time.time() - time_item_start
        time_list.append(time_)

        result_image_path = save_path + image_name[0:-4] + '/'
        if not os.path.exists(result_image_path):
            os.makedirs(result_image_path)

        if is_single:
            view_n_central = view_n // 2
            print('PSNR: {:.4f}, SSIM: {:.4f}, TIME: {:.4f}'.format(psnr_image[view_n_central, view_n_central],
                                                                    ssim_image[view_n_central, view_n_central], time_))

            hr_y_item = np.clip(hr_y[view_n_central, view_n_central, :, :] * 255.0, 16.0, 235.0)
            hr_y_item = hr_y_item[:, :, np.newaxis]
            hr_cb_item = hr_cbcr[0, 0, :, :, 1:2]
            hr_cr_item = hr_cbcr[0, 0, :, :, 0:1]
            hr_ycbcr_item = np.concatenate((hr_y_item, hr_cb_item, hr_cr_item), 2)
            hr_rgb_item = color.ycbcr2rgb(hr_ycbcr_item) * 255.0
            img_save_path = result_image_path + 'central.png'
            cv2.imwrite(img_save_path, hr_rgb_item)

            psnr_ = psnr_image[view_n_central, view_n_central]
            psnr_list.append(psnr_)
            ssim_ = ssim_image[view_n_central, view_n_central]
            ssim_list.append(ssim_)
        else:
            for i in range(view_n):
                for j in range(view_n):
                    print('{:6.4f}/{:6.4f}'.format(psnr_image[i, j], ssim_image[i, j]), end='\t\t')
                print('')

            print(
                'PSNR Avr: {:.4f}, Max: {:.4f}, Min: {:.4f}, SSIM: Avr: {:.4f}, Max: {:.4f}, Min: {:.4f}, TIME: {:.4f}'
                    .format(np.mean(psnr_image), np.max(psnr_image), np.min(psnr_image),
                            np.mean(ssim_image), np.max(ssim_image), np.min(ssim_image), time_))

            for i in range(view_n):
                for j in range(view_n):
                    hr_y_item = np.clip(hr_y[i, j, :, :] * 255.0, 16.0, 235.0)
                    hr_y_item = hr_y_item[:, :, np.newaxis]
                    hr_cb_item = hr_cbcr[i, j, :, :, 1:2]
                    hr_cr_item = hr_cbcr[i, j, :, :, 0:1]
                    hr_ycbcr_item = np.concatenate((hr_y_item, hr_cb_item, hr_cr_item), 2)
                    hr_rgb_item = color.ycbcr2rgb(hr_ycbcr_item) * 255.0
                    img_save_path = result_image_path + str(i) + str(j) + '.png'
                    cv2.imwrite(img_save_path, hr_rgb_item)

            psnr_ = np.mean(psnr_image)
            psnr_list.append(psnr_)
            ssim_ = np.mean(ssim_image)
            ssim_list.append(ssim_)

        xls_list.append([image_name, psnr_, ssim_, time_])

    xls_list.append(['average', np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)])
    xls_list = np.array(xls_list)

    result = pd.DataFrame(xls_list, columns=['image', 'psnr', 'ssim', 'time'])
    result.to_csv(save_path + 'result.csv')

    print('-' * 100)
    print('AVR: PSNR: {:.4f}, SSIM: {:.4f}, TIME: {:.4f}'.format(np.mean(psnr_list), np.mean(ssim_list),
                                                                 np.mean(time_list)))
    print('all done')


def predict_y(lr_y, gt_hr_y, view_n, is_single, model_dic, scale):
    """
    perdict channel Y
    :param lr_y:
    :param gt_hr_y:
    :param view_n:
    :param model: tuple of model
    :return:
    """
    lr_y, gt_hr_y = torch.from_numpy(lr_y.copy()), torch.from_numpy(gt_hr_y.copy())
    torch.no_grad()

    psnr_image = np.zeros((view_n, view_n))
    ssim_image = np.zeros((view_n, view_n))
    image_h = gt_hr_y.shape[0] // view_n
    image_w = gt_hr_y.shape[1] // view_n
    hr_y = np.zeros((view_n, view_n, image_h, image_w), dtype=np.float32)

    # model reading
    if is_single:
        model = model_dic['single']
        # for central image
        u_list = [view_n // 2]
        v_list = [view_n // 2]
        psnr_image, ssim_image, hr_y = test_all(lr_y, gt_hr_y, view_n, u_list, v_list, model, psnr_image, ssim_image,
                                                hr_y, scale)
        return psnr_image, ssim_image, hr_y

    uv_dic = uv_list_by_n(view_n)

    for item in range(3, view_n + 1, 2):
        if item == 3:
            model = model_dic['3h']
            u_list = uv_dic['u3h']
            v_list = uv_dic['v3h']
            psnr_image, ssim_image, hr_y = test_all(lr_y, gt_hr_y, view_n, u_list, v_list, model,
                                                    psnr_image, ssim_image, hr_y, scale)

            model = model_dic['3v']
            u_list = uv_dic['u3v']
            v_list = uv_dic['v3v']
            psnr_image, ssim_image, hr_y = test_all(lr_y, gt_hr_y, view_n, u_list, v_list, model,
                                                    psnr_image, ssim_image, hr_y, scale)

            model = model_dic['3hv']
            u_list = uv_dic['u3hv']
            v_list = uv_dic['v3hv']
            psnr_image, ssim_image, hr_y = test_all(lr_y, gt_hr_y, view_n, u_list, v_list, model,
                                                    psnr_image, ssim_image, hr_y, scale)
        model = model_dic[str(item)]
        u_list = uv_dic['u' + str(item)]
        v_list = uv_dic['v' + str(item)]
        psnr_image, ssim_image, hr_y = test_all(lr_y, gt_hr_y, view_n, u_list, v_list, model,
                                                psnr_image, ssim_image, hr_y, scale)

    return psnr_image, ssim_image, hr_y


def predict_cbcr(lr_cbcr, scale, view_n, is_single=True):
    if is_single:
        hr_cbcr = np.zeros((1, 1, lr_cbcr.shape[0] // view_n * scale, lr_cbcr.shape[1] // view_n * scale, 2))
        image_bicubic = cv2.resize(lr_cbcr[view_n // scale::view_n, view_n // scale::view_n, :],
                                   (hr_cbcr.shape[3], hr_cbcr.shape[2]),
                                   interpolation=cv2.INTER_CUBIC)
        hr_cbcr[0, 0, :, :, :] = image_bicubic
        return hr_cbcr

    hr_cbcr = np.zeros((view_n, view_n, lr_cbcr.shape[0] // view_n * scale, lr_cbcr.shape[1] // view_n * scale, 2))
    for i in range(view_n):
        for j in range(view_n):
            image_bicubic = cv2.resize(lr_cbcr[i::view_n, j::view_n, :], (hr_cbcr.shape[3], hr_cbcr.shape[2]),
                                       interpolation=cv2.INTER_CUBIC)
            hr_cbcr[i, j, :, :, :] = image_bicubic
    return hr_cbcr


def resLF_model_reading_single(model_path, view_n, scale, interpolation):
    model = {}
    model_item = resLF(n_view=view_n, scale=scale)
    model_item.cuda()
    state_dict = torch.load(model_path + '{}_{}_{}.pkl'.format(scale, view_n, interpolation))
    model_item.load_state_dict(state_dict)
    model['single'] = model_item

    return model


def resLF_model_reading_all(model_path, view_n, scale, interpolation):
    model_dic = {}
    for item in range(3, view_n + 1, 2):
        task = ['3h', '3v', '3hv']
        if item == 3:
            for i in task:
                model = resLF(n_view=item, scale=scale)
                model.cuda()
                state_dict = torch.load(model_path + '{}_{}_{}.pkl'.format(scale, i, interpolation))
                model.load_state_dict(state_dict)
                model_dic[i] = model

        model = resLF(n_view=item, scale=scale)
        model.cuda()
        state_dict = torch.load(model_path + '{}_{}_{}.pkl'.format(scale, item, interpolation))
        model.load_state_dict(state_dict)
        model_dic[str(item)] = model

    return model_dic


def test_all(test_image, gt_image, view_num_all, u_list, v_list, model, psnr_image, ssim_image, pre_lf, scale):
    for i in range(0, len(u_list), 1):
        u = u_list[i]
        v = v_list[i]

        model.eval()
        train_data_0, train_data_90, train_data_45, train_data_135, gt_data = \
            multi_input_all(test_image, gt_image, view_num_all, u, v, scale)

        train_data_0, train_data_90, train_data_45, train_data_135, gt_data = \
            train_data_0.cuda(), train_data_90.cuda(), train_data_45.cuda(), train_data_135.cuda(), gt_data.cuda()

        # Forward pass: Compute predicted y by passing x to the model
        with torch.no_grad():
            gt_pred = model(train_data_0, train_data_90, train_data_45, train_data_135)

        # calculate the PSNR and SSIM values
        output = gt_pred[0, 0, :, :]
        img_pre = output.cpu().numpy()
        img_pre = np.clip(img_pre, 16/255, 235/255)

        output = gt_data[0, :, :]
        gt_img = output.cpu().numpy()
        image_h = gt_img.shape[0]
        image_w = gt_img.shape[1]

        compare_loss = (img_pre - gt_img) ** 2
        compare_loss = compare_loss.sum() / (image_w * image_h)
        psnr = 10 * log10(1 / compare_loss)
        ssim = compare_ssim(img_pre, gt_img)

        psnr_image[u, v] = psnr
        ssim_image[u, v] = ssim
        pre_lf[u, v, :, :] = img_pre

    return psnr_image, ssim_image, pre_lf


if __name__ == '__main__':
    parser = opts_parser()
    args = parser.parse_args()

    image_path = args.image_path
    model_path = args.model_path
    save_path = args.save_path
    view_n_ori = args.original_length
    view_n = args.crop_length
    is_single = args.is_single
    scale = args.scale
    interpolation = args.interpolation
    gpu_no = args.gpu_no

    main(image_path=image_path,
         model_path=model_path,
         save_path=save_path,
         view_n_ori=view_n_ori,
         view_n=view_n,
         scale=scale,
         is_single=is_single,
         interpolation=interpolation,
         gpu_no=gpu_no)
