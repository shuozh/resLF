# resLF
Residual Networks for Light Field Image Super-Resolution

CONTACT: [Shuo Zhang](https://shuozh.github.io/)  
(zhangshuo@bjtu.edu.cn)

Any scientific work that makes use of our code should appropriately mention this in the text and cite our CVPR 2019 paper. For commercial use, please contact us.

### PAPER TO CITE:

Shuo Zhang, Youfang Lin, Hao Sheng.  
Residual Networks for Light Field Image Super-Resolution, 
IEEE Conference of Computer Vision and Pattern Recognition, 2019

### How to use:

**Test example:** 

- For central view:
  
        python resLF_test.py -I image_test_path/ -M model_path/ -S save_path/ -o 9 -c 3 -g 0 -s 2 -i blur -C y

- For full Light Field SR ($7\times7$):

        python resLF_test.py -I image_test_path/ -M model_path/ -S save_path/ -o 9 -c 7 -g 0 -s 2 -i blur -C n

- For Help
        
        python resLF_test.py -h
    

### Note:

We provide different downsampling methods named 'Bicubic' and 'Blur'. 
In 'Blur', the sub-aperture images are first blurred using the normalized box filter (the filter size is $scale \times scale$), then regularly decimated to the desired resolution. 
In 'Bicubic', the sub-aperture images are directly downsampled using bicubic downsampling method.
Note that 'Blur' downsampling method achieves better results compared with 'Bicubic' downsampling method.

### Dataset:
The training and testing example dataset can be found at: http://lightfields.stanford.edu/ (Lytro Illum), 
https://mmspg.epfl.ch/downloads/epfl-light-field-image-dataset/ (Lytro Illum) and  https://lightfield-analysis.uni-konstanz.de (Synthesis Light Field).


### Envs:

python 3.6.5

pytorch 0.4.0

cuda 10.1

cudnn 7.5.1


### Time log:

2019.09.19 Another downsampling method is provided. 
2019.06.12 The package released.


