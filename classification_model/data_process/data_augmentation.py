#!/usr/bin/python
# coding:utf-8
# 娑婆.南瞻部洲
# Amitabha.ma

# 般若波罗蜜多心经
# 观自在菩萨,行深般若波罗蜜多时,照见五蕴皆空,度一切苦厄,舍利子,色不异空,空不异色,
# 色即是空，空即是色，受想行识，亦复如是。舍利子，是诸法空相，不生不灭，不垢不净，不增不减，
# 是故空中无色，无受想行识，无眼耳鼻舌身意，无色声香味触发，无眼界，乃至无异世界，无无明，
# 亦无无明尽，乃至无老死，亦无老死尽，无苦集灭道，无智亦无得，以无所得故，菩提萨埵，依般若波罗蜜多故，
# 心无挂碍，无挂碍故，无有恐怖，远离颠倒梦想，究竟涅槃，三世诸佛，依般若波罗蜜多故。得阿耨多罗三藐三菩提。
# 故知波热波罗蜜多，是大神咒，是大明咒，是无上咒，是无等等咒，能除一切苦，真实不虚。故说般若波罗蜜多咒，
# 即说咒曰，揭谛揭谛，波罗揭谛，波罗僧揭谛，菩提萨婆诃

# 往生净土神咒
# 拔一切业障根本得生净土陀罗尼
# 南无阿弥多婆夜。哆他伽多夜。哆地夜他。
# 阿弥利都婆毗。阿弥利哆。悉耽婆毗。
# 阿弥唎哆。毗迦兰帝。阿弥唎哆。毗迦兰多。
# 伽弥腻。伽伽那。枳多迦利。娑婆诃。
# 南无阿弥陀佛，南无阿弥陀佛，南无阿弥陀佛
import cv2 as cv
from imgaug import augmenters as iaa
import random


def rotate_img(img,angle,center=None,scale=1.0):
    h,w,c = img.shape
    if center is None:
        center = (w / 2, h / 2 )
    m = cv.getRotationMatrix2D(center,angle,scale)
    rotate = cv.warpAffine(img,m,(w,h))
    return rotate


def crop_img(img):
    seq = iaa.Sequential([
        iaa.Crop(px=(int(random.uniform(0,100)),int(random.uniform(0,100)))),
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0,3.0))
    ])

    #返回的类似img的对象
    return seq.augment_image(img)


def gaussian_blur(img):
    seq = iaa.Sequential([
        iaa.AdditiveGaussianNoise(loc=0,scale=(0.0,random.uniform(0,0.05) * 255),
                                  per_channel=random.uniform(0,0.5))
    ])
    return seq.augment_image(img)


def add_to_hue_and_saturation(img):
    seq = iaa.Sequential([iaa.AddToHueAndSaturation((-20,20))])
    return seq.augment_image(img)


def contrast_normalization(img):
    seq = iaa.Sequential([iaa.ContrastNormalization((0.5,2.0),per_channel=0.5)])
    return seq.augment_image(img)


def gray_scale(img):
    seq = iaa.Sequential([iaa.Grayscale(alpha=(0.0, 1.0))])
    return seq.augment_image(img)


def salt(img):
    seq = iaa.Sequential([iaa.Salt(per_channel=True,p=random.uniform(0,0.5))])
    return seq.augment_image(img)


def elastic_transformation(img):
    seq = iaa.Sequential([iaa.ElasticTransformation(alpha=(random.uniform(0,0.5),
                                                           random.uniform(3,3.5)),sigma=0.25)])
    return seq.augment_image(img)


def piecewise_affine(img):
    seq = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01,0.05))])
    return seq.augment_image(img)


def perspective_transform(img):
    seq = iaa.Sequential([iaa.PerspectiveTransform(scale=(0.01,0.1))])
    return seq.augment_image(img)


def gaussian(img):
    seq = iaa.Sequential([iaa.Fliplr(0.5),iaa.GaussianBlur((0,3.0))])
    return seq.augment_image(img)


def averageblur(img):
    seq = iaa.Sequential([iaa.AverageBlur((0,3.0))])
    return seq.augment_image(img)


def medianblur(img):
    seq = iaa.Sequential([iaa.AverageBlur((0,3.0))])
    return seq.augment_image(img)

def sharpen(img):
    seq = iaa.Sequential([iaa.Sharpen(alpha=(0,1.0),lightness=(0.75,1.5))])
    return seq.augment_image(img)

def emboss(img):
    seq = iaa.Sequential([iaa.Emboss(alpha=(0,1.0),strength=(0,2.0))])
    return seq.augment_image(img)

def dropout(img):
    seq = iaa.Sequential([iaa.Dropout((0.01,0.1),per_channel=random.uniform(0,0.5))])
    return seq.augment_image(img)

def edge_detect(img):
    seq = iaa.Sequential([iaa.EdgeDetect(alpha=(0.5,1.0))])
    return seq.augment_image(img)

def affine(img):
    seq = iaa.Sequential([iaa.GaussianBlur((0,3.0))],iaa.Affine(translate_px={'x':(-40,40)}))
    return seq.augment_image(img)

def aug(img):
    seq = iaa.WithChannels(
        channels=[0,1],
        children=iaa.Add((-30,30))
    )
    return seq.augment_image(img)

'''
for example
img_path= ''
img = cv.imread(img_path)
auged = aug(img)
cv.imwrite('example.jpg',auged)

'''