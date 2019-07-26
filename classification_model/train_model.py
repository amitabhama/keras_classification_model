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
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import os
from model.model import inceptionresnetv2_model,inceptionv3_model
from model.model import vgg_model,mobilenet_model,nasnetlarge_model
from model.model import densenet,resnet50_model,xception_model


filepath = os.path.dirname(__file__)

'''
调节参数
'''
lr = 0.0001
batch_size = 32
epoch = 10
train_data_dir = ''
val_data_dir = ''
class_num = ''

# 模型需要的图像大小,通常宽和高相等
img_width = 224
img_height = 224


model = densenet(lr,class_num,'densenet201',224,224,-5,epoch,'sgd')

l_train_num = []
for r,dr,files in os.walk(train_data_dir):
    for fil in files:
        img = os.path.join(r,fil)
        l_train_num.append(img)

train_data_samples = len(l_train_num) # 训练的样本数据量

l_val_num = []
for r, dr,files in os.walk(val_data_dir):
    for fil in files:
        img = os.path.join(r, fil)
        l_val_num.append(img)

val_data_samples = len(l_val_num) #验证的样本数据量


classes = os.listdir(train_data_dir) #获取分类类别数目类别名称

best_model_file_path = os.path.join(filepath,'weights.h5') #当前路径保存模型

best_model = ModelCheckpoint(best_model_file_path,monitor='val_acc',verbose=1,save_best_only=True)

# 增加训练集图片的增强器
train_data_generation = ImageDataGenerator(
    rescale= 1./ 255,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=10.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_data_generation = ImageDataGenerator(rescale=1.0/255)

train_generator = train_data_generation.flow_from_directory(
    train_data_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    shuffle=True,
    classes=classes,
    class_mode='categorical'
)

val_generator = val_data_generation.flow_from_directory(
    val_data_dir,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    shuffle=True,
    classes=classes,
    class_mode='categorical'
)


model.fit_generator(
    train_generator,
    steps_per_epoch = int(train_data_samples / epoch),
    epochs=epoch,
    validation_data= val_generator,
    validation_steps= int(val_data_samples / epoch),
    callbacks=[best_model],
)