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
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense,Dropout,Flatten
from keras.applications.mobilenet import MobileNet
import os
from keras.models import Model,load_model
from keras.optimizers import SGD,Adam,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetLarge
from keras.applications.densenet import DenseNet121,DenseNet169,DenseNet201
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import tensorflow as tf
# from keras.utils.multi_gpu_utils import multi_gpu_model

os.environ['CUDA_VISIBEL_DEVICES'] = '0'

options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(options=options)
sess = tf.Session(config=config)

'''
12个模型
resnet50
mobilenet_v1
mobilenet_v2
xception
nasnetlarge
densenet121,densenet169,densenet201,
inceptionv3
vgg16,vgg19
inceptionresnetv2
'''

def inceptionresnetv2_model(lr,class_num,img_rows=224,img_cols=224,frozen_layer_index=-1,epoch=5,opt='sgd'):
    resnetv2_model = InceptionResNetV2(includ_top=False,input_tensor=None,input_shape=(img_cols,img_rows,3),
                                       weights='imagenet',classes=class_num)
    for layer in resnetv2_model.layers[:frozen_layer_index]:
        layer.trainable = False

    x = resnetv2_model.outputs
    x = Dropout(0.7)(x)
    x = Flatten()(x)
    predictions = Dense(units=class_num,activation='softmax')(x)

    if epoch >= 5:
        if epoch % 5 == 0:
            lr = lr * 0.1
    else:
        lr = lr

    model = Model(inputs=resnetv2_model.inputs,outputs=predictions)
    optimizers = get_optimizer(lr,opt)
    model.summary()
    model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def vgg_model(lr,class_num,name='vgg16',img_rows=224,img_cols=224,frozen_layer_index=-1,epoch=5,opt='sgd'):
    if name == 'vgg16':
        v_model = VGG16(include_top=False,input_tensor=None,input_shape=(img_cols,img_rows,3),
                        weights='imagenet',classes=class_num)
    elif nam == 'vgg19':
        v_model = VGG19(include_top=False,input_tensor=None,input_shape=(img_cols,img_rows,3),
                        weights='imagenet',classes=class_num)
    else:
        raise ValueError('请输入vgg16或者vgg19')

    for layer in v_model.layers[:frozen_layer_index]:
        layer.trainable = False

    x = v_model.outputs
    x = Dropout(0.7)(x)
    x = Flatten()(x)
    predictions = Dense(units=class_num,activation='softmax')(x)

    if epoch >= 5:
        if epoch % 5 == 0:
            lr = lr *0.1
    else:
        lr = lr

    model = Model(inputs=v_model.inputs,outputs=predictions)
    optimizers = get_optimizer(lr,opt)
    model.summary()
    model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def inceptionv3_model(lr,class_num,img_rows=299,img_cols=299,frozen_layer_index=-1,epoch=5,opt='adam'):
    inception_model = InceptionV3(include_top=False,input_tensor=None,input_shape=(img_rows,img_cols,3),
                                  weights='imagenet',classes=class_num)
    for layer in inception_model.layers[:frozen_layer_index]:
        layer.trainable = False

    x = inception_model.outputs
    x = Dropout(0.7)(x)
    x = Flatten()(x)
    predictions = Dense(units=class_num,activation='softmax')(x)

    if epoch >= 5:
        if epoch % 5 == 0:
            lr = lr *0.1
    else:
        lr = lr

    model = Model(inputs=inception_model.inputs,outputs=predictions)
    optimizers = get_optimizer(lr,opt)
    model.summary()
    model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def nasnetlarge_model(lr,class_num,img_rows=299,img_cols=299,frozen_layer_index=-1,epoch=5,opt='adam'):
    nasnet_model = NASNetLarge(includ_top=False,input_tensor=None,input_shape=(img_rows,img_cols,3),
                               weights='imagenet',classes=class_num)
    for layer in nasnet_model.layers[:frozen_layer_index]:
        layer.trainable = False

    x = nasnet_model.outputs
    x = Dropout(0.7)(x)
    x = Flatten()(x)
    predictions = Dense(units=class_num,activation='softmax')(x)

    if epoch >= 5:
        if epoch % 5 == 0:
            lr = lr *0.1
    else:
        lr = lr

    model = Model(inputs=nasnet_model.inputs,outputs=predictions)
    optimizers = get_optimizer(lr,opt)
    model.summary()
    model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def xception_model(lr,class_num,img_rows=299,img_cols=299,frozen_layer_index=-1,epoch=5,opt='adam'):
    x_model = Xception(include_top=False,input_tensor=None,input_shape=(img_rows,img_cols,3),
                       weights='imagenet',classes=class_num)
    for layer in x_model.layers[:frozen_layer_index]:
        layer.trainable = False

    x = x_model.outputs
    x = Dropout(0.7)(x)
    x = Flatten()(x)
    predictions = Dense(units=class_num,activation='softmax')(x)
    if epoch >= 5:
        if epoch % 5 == 0:
            lr = lr * 0.1
    else:
        lr = lr
    model = Model(inputs=x_model.inputs,outputs=predictions)
    optimizers = get_optimizer(lr,opt)
    model.summary()
    model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['accuracy'])

    return model



def get_optimizer(lr,opt='sgd'):
    '''
    :param lr: 学习率
    :param opt: 优化器名称
    :return: 优化器
    '''
    if opt == 'sgd':
        opt_s = SGD(lr=lr, decay=1e-6,momentum=0.9,nasterov=True)
    elif opt == 'adam':
        opt_s = Adam(lr=lr,decay=1e-6)
    elif opt == 'rmsprop':
        opt_s = RMSprop(lr=lr,decay=1e-6)
    else:
        return 'please set optmizier name , sgd or adam or rmsprop'
    return opt_s


def resnet50_model(lr,class_num,img_rows=224,img_cols=224,frozen_layer_index=-1,epoch=5,opt='sgd',):
    '''
    :param lr: 学习率
    :param class_num: 分类的类别数目
    :param img_rows: 图片的行数
    :param img_cols: 图片的列数
    :param trainable_layer_index: 冻结训练的层索引
    :param epoch: 训练集的整体轮次数
    :param opt: 优化器名称
    :return:
    '''
    resnet50 = ResNet50(input_tensor=None,include_top=False, input_shape=(img_cols,img_rows,3)
                        ,weights='imagenet',classes=class_num)
    for layer in resnet50.layers[:frozen_layer_index]:
        layer.trainable = False

    x = resnet50.output
    x = Dropout(rate=0.7)(x)
    x = Flatten()(x)

    predictions = Dense(units=class_num,activation='softmax')(x)

    model = Model(inputs=resnet50.input,outputs=predictions)
    if epoch >= 5:
    # 每5轮学习率降低10倍
        if epoch % 5 ==0:
            lr = lr*0.1
    else:
        lr = lr
    print('model will train {} epoches'.format(epoch))
    optimizers = get_optimizer(lr=lr,opt=opt)
    model.summary()
    model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def mobilenet_model(lr, class_num,img_rows=224,img_cols=224,frozen_layer_index=-1,epoch=5,opt='adam'):
    '''
    :param lr: 学习率
    :param class_num: 分类的类别数目
    :param img_rows:  图片的行数
    :param img_cols:  图片的列数
    :param trainable_layer_index: 冻结训练层的索引数
    :param epoch: 训练的论次数
    :param opt: 优化器名称
    :return:
    '''

    mobilenet = MobileNet(input_shape=(img_rows,img_cols,3),input_tensor=False, include_top=False,
                          weights='imagenet', classes=class_num)
    for layer in mobilenet.layers[:frozen_layer_index]:
        layer.trainable = False

    x = mobilenet.output
    x = Dropout(rate=0.7)(x)
    x = Flatten()(x)
    predictions = Dense(units=class_num,activation='sorfmax')(x)

    model = Model(inputs=mobilenet.inputs, outputs=predictions)
    if epoch > 5:
    # 每5轮学习率降低10倍
        if epoch % 5 ==0:
            lr = lr*0.1
    else:
        lr = lr
    print('model will train {} epoches'.format(epoch))
    optimizers = get_optimizer(lr=lr,opt=opt)
    model.summary()
    model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def mobilenet_v2_model(lr, class_num,img_rows=224,img_cols=224,frozen_layer_index=-1,epoch=5,opt='rmsprop'):
    '''
    :param lr:  学习率
    :param class_num: 需要分类的类别数
    :param img_rows:  图片的行数
    :param img_cols:  图片的列数
    :param trainable_layer_index: 冻结训练层的索引
    :param epoch: 训练的轮次数
    :param opt: 优化器名称
    :return:
    '''


    v2_model = MobileNetV2(includ_top=False,input_tensor=None, input_shape=(img_cols,img_rows,3),
                           weights='imagenet',classes=class_num)
    for layer in v2_model.layers[:frozen_layer_index]:
        layer.trainable = False

    x = v2_model.output
    x = Dropout(rate=0.7)(x)
    x = Flatten()(x)
    predictions = Dense(units=class_num, activation='softmax')(x)

    model = Model(inputs=v2_model.inputs, outputs=predictions)
    if epoch >= 5:
        # 每5轮学习率降低10倍
        if epoch % 5 == 0:
            lr = lr * 0.1
    else:
        lr = lr
    print('model will train {} epoches '.format(epoch))
    optimizers = get_optimizer(lr=lr, opt=opt)
    model.summary()
    model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def densenet(lr, class_num,name='densenet121',img_rows=224,img_cols=224,frozen_layer_index=-1,epoch=5,opt='adam'):
    if name == 'densenet121':
        dense_model = DenseNet121(includ_top=False,input_tensor=None,input_shape=(img_rows,img_cols,3),
                                  weight='imagenet',classes=class_num)
    elif name == 'densenet169':
        dense_model = DenseNet169(includ_top=False,input_tensor=None,input_shape=(img_rows,img_cols,3),
                                  weight='imagenet',classes=class_num)
    elif name == 'densenet201':
        dense_model = DenseNet201(includ_top=False,input_tensor=None,input_shape=(img_rows,img_cols,3),
                                  weight='imagenet',classes=class_num)
    else:
        raise ValueError ('输入densenet121或者densenet169或者densenet201')

    for layer in dense_model.layers[:frozen_layer_index]:
        layer.trainable = False
    x = dense_model.output
    x = Dropout(0.7)(x)
    x = Flatten()(x)
    predictions = Dense(units=class_num,activation='softmax')(x)
    model = Model(inputs=dense_model.inputs,outputs=predictions)
    if epoch >= 5:
        if epoch %5 == 0:
            lr = lr *0.1
    else:
        lr = lr

    optimizers = get_optimizer(lr, opt)
    model.compile(optimizers,loss='categorical_crossentropy', metrics=['accuracy'])
    return model





