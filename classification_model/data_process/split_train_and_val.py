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
import os
import shutil

def get_train_and_val_data(all_image_path,target_path, train_split_num=0.95):
    l = []
    # 创建一个目标文件夹
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    # 获取所有图像文件,并创建对应名称目录
    for r,dr,files in os.walk(all_image_path):
        for fil in files:
            img_path = os.path.join(r,fil)
            class_name = img_path.split('/')[-2]
            if not os.path.exists(os.path.join(target_path,class_name)):
                os.mkdir(class_name)
            l.append(img_path)

    # 获取所欲图片数量
    all_img_num = len(l)
    train_img = l[:int(train_split_num * all_img_num)] # 获取全部图像的95%作为训练集
    val_img = l[int(train_split_num * all_img_num):] # 剩下的5%作为验证集

    # 复制训练集
    for i in train_img:
        class_name = i.split('/')[-2]
        image_name = i.split('/')[-1]
        shutil.copy(i,os.path.join(target_path,class_name,image_name))
        print('正在复制训练集图片 从{} 复制到 {}'.format(i, os.path.join(target_path,class_name,image_name)))

    # 复制验证集
    for i in val_img:
        class_name = i.split('/')[-2]
        image_name = i.split('/')[-1]
        shutil.copy(i,os.path.join(target_path,class_name,image_name))
        print('正在复制测试集图片 从{} 复制到 {}'.format(i, os.path.join(target_path,class_name,image_name)))
