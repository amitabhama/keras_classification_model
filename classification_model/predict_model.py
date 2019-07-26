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
from keras.models import load_model
import numpy as np
import os
from train_model import best_model_file_path
from train_model import batch_size,img_width,img_height,classes
from keras.preprocessing.image import ImageDataGenerator
os.environ['CUDA_VISIBEL_DEVICES'] = '0' #指定gpu卡号

test_img_dir = ''
l_test_data = []
for r,dr,files in os.walk(test_img_dir):
    for fil in files:
        test_img = os.path.join(r,fil)
        l_test_data.append(test_img)

test_data_number = len(l_test_data)
augmentation_number = 5

model = load_model(best_model_file_path)
test_data_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

for index in range(augmentation_number):
    print('第{}次测试增强的图片'.format(index))
    random_seed = np.random.random_integers(0,100000)
    test_generator = test_data_generator.flow_from_directory(
        test_img_dir,
        target_size=(img_height,img_width),
        batch_size=batch_size,
        seed=random_seed,
        class_mode=None,
        classes=None
    )
    test_img_list = test_generator.filenames
    if index == 0:
        predictions = model.predict_generator(test_generator,test_data_number)
    else:
        predictions += + model.predict_generator(test_generator,test_data_number)

predictions /= augmentation_number


with open('result.csv','a+') as f:
    for i in classes:
        f.write(i+',')
    f.write('\n')
    for index,value in enumerate(test_img_list):
        pred = ['%.6f' % p for p in predictions[index:]]
        if index % 100 == 0:
            print('{} / {}'.format(index,test_data_number))
        f.write('%s,%s\n'%(os.path.basename(value),','.join(pred)))