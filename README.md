### pytorch Implementation of U-Net, R2U-Net, Attention U-Net, Attention R2U-Net

**(This repository is no longer being updated)**

**U-Net: Convolutional Networks for Biomedical Image Segmentation**

https://arxiv.org/abs/1505.04597

**Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation**

https://arxiv.org/abs/1802.06955

**Attention U-Net: Learning Where to Look for the Pancreas**

https://arxiv.org/abs/1804.03999

**Attention R2U-Net : Just integration of two recent advanced works (R2U-Net + Attention U-Net)**


## U-Net
![U-Net](/img/U-Net.png)


## R2U-Net
![R2U-Net](/img/R2U-Net.png)

## Attention U-Net
![AttU-Net](/img/AttU-Net.png)

## Attention R2U-Net
![AttR2U-Net](/img/AttR2U-Net.png)

## Evaluation
we just test the models with [ISIC 2018 dataset](https://challenge2018.isic-archive.com/task1/training/). The dataset was split into three subsets, training set, validation set, and test set, which the proportion is 70%, 10% and 20% of the whole dataset, respectively. The entire dataset contains 2594 images where 1815 images were used
for training, 259 for validation and 520 for testing models.

![evaluation](/img/Evaluation.png)
