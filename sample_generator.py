# -*- coding:utf8 -*-
"""
功能：将语料中的每行文本绘制成图像
输入：语料文件
输出：文本图像
参数：在config.cfg.py中: config_path, corpus, dict, FONT_PATH
"""
import codecs
import numpy as np
import TextImageGenerator.config.cfg as cfg
import os
from PIL import Image, ImageDraw, ImageFont
import progressbar
import glob
import cv2
from colormath.color_objects import CMYKColor,sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


def rotate_bound(image, angle, bg_color):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=bg_color)

def distort(input_img):
    h, w, d = input_img.shape
    rotated = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1)
    return rotated


class TextGenerator():
    def __init__(self, save_path):
        """初始化参数来自config文件
        """
        # 语料参数
        self.max_row_len = cfg.max_row_len
        self.max_label_len = cfg.max_label_len  # CTC最大输入长度
        self.n_samples = cfg.n_samples
        self.dictfile = cfg.dict  # 字典
        self.dict = []
        self.corpus_file = cfg.corpus  # 语料集
        self.save_path = save_path
        # 加载字体文件
        self.font_factor = 1  # 控制字体大小
        # 加载字体文件
        self.load_fonts()
        # 加载语料集
        self.build_dict()
        self.build_train_list(self.n_samples, self.max_row_len)


    def load_fonts(self):
        """ 加载字体文件并设定字体大小
            TODO： 无需设定字体大小，交给pillow
        :return: self.fonts
        """
        self.fonts = {}  # 所有字体
        self.font_name = []  # 字体名，用于索引self.fonts
        # 字体完整路径
        font_path = os.path.join(cfg.FONT_PATH, "*.*")
        # 获取全部字体路径，存成list
        fonts = list(glob.glob(font_path))
        # 遍历字体文件
        for each in fonts:
            # 字体大小
            fonts_list = {}  # 每一种字体的不同大小
            font_name = each.split('\\')[-1].split('.')[0]  # 字体名
            self.font_name.append(font_name)
            font_size = 60
            for j in range(0, 10):  # 当前字体的不同大小
                # 调整字体大小
                cur_font = ImageFont.truetype(each, font_size, 0)
                fonts_list[str(j)] = cur_font
                font_size -= 2
            self.fonts[font_name] = fonts_list

    def build_dict(self):
        """ 打开字典，加载全部字符到list
            每行是一个字
        :return: self.dict
        """
        with codecs.open(self.dictfile, mode='r', encoding='utf-8') as f:
            # 按行读取语料
            for line in f:
                # 当前行单词去除结尾
                word = line.strip('\r\n')
                # 只要没超出上限就继续添加单词
                self.dict.append(word)
        # 最后一类作为空白占位符
        self.blank_label = len(self.dict)

    def mapping_list(self, DATASET_DIR):
        # 写图像文件名和类别序列的对照表
        file_path = os.path.join(DATASET_DIR, 'map_list.txt')
        with codecs.open(file_path, mode='w', encoding='utf-8') as f:
            for i in range(len(self.train_list)):
                f.write("{}.jpeg {} \n".format(i, self.train_list[i]))

    def build_train_list(self, n_samples, max_row_len=None):
        # 过滤语料，留下适合的内容组成训练list
        print('正在加载语料...')
        assert max_row_len <= self.max_label_len  # 最大类别序列长度
        self.n_samples = n_samples  # 语料总行数
        sentence_list = []  # 存放每行句子
        self.train_list = []
        self.label_len = [0] * self.n_samples  # 类别序列长度
        self.label_sequence = np.ones([self.n_samples, self.max_label_len]) * -1  # 类别ID序列

        with codecs.open(self.corpus_file, mode='r', encoding='utf-8') as f:
            # 按行读取语料
            for sentence in f:
                sentence = sentence.strip()  # 去除行末回车
                if len(sentence_list) < n_samples:
                    # 只要长度和数量没超出上限就继续添加单词
                    sentence_list.append(sentence)

        np.random.shuffle(sentence_list)  # 打乱语料
        if len(sentence_list) < self.n_samples:
            raise IOError('语料不足')

        # 遍历语料中的每一句(行)
        for i, sentence in enumerate(sentence_list):
            # 每个句子的长度
            label_len = len(sentence)
            filted_sentence = ''
            # 将单词分成字符，然后找到每个字符对应的整数ID list
            # n_samples个样本每个一行max_row_len元向量(单词最大长度)，每一元为该字符的整数ID
            label_sequence = []
            for j, word in enumerate(sentence):
                index = self.dict.index(word)
                label_sequence.append(index)
                filted_sentence += word

            if filted_sentence is not '':
                # 当前样本的类别序列及其长度
                self.label_len[i] = label_len
                self.label_sequence[i, 0:self.label_len[i]] = label_sequence
            else:  # 单独处理空白样本
                self.label_len[i] = 1
                self.label_sequence[i, 0:self.label_len[i]] = self.blank_label  # 空白符

        self.label_sequence = self.label_sequence.astype('int')
        self.train_list = sentence_list  # 过滤后的训练集
        self.mapping_list(self.save_path)  # 保存图片名和类别序列的 map list

    def paint_text(self, text, i):
        """ 使用PIL绘制文本图像，传入画布尺寸，返回文本图像
        :param h: 画布高度
        :param w: 画布宽度
        :return: img
        """
        # 创建画布背景
        bg_b = np.random.randint(0, 255)  # 背景色
        bg_g = np.random.randint(0, 255)
        bg_r = np.random.randint(0, 255)
        # 前景色
        fg_b = np.random.randint(0, 255)  # 背景色
        fg_g = np.random.randint(0, 255)
        fg_r = np.random.randint(0, 255)
        # 计算前景和背景的彩色相似度
        bg_color = sRGBColor(bg_b, bg_g, bg_r)
        bg_color = convert_color(bg_color, CMYKColor)  # 转cmyk
        bg_color = convert_color(bg_color, LabColor)
        fg_color = sRGBColor(fg_b, fg_g, fg_r)
        fg_color = convert_color(fg_color, CMYKColor)  # 转cmyk
        fg_color = convert_color(fg_color, LabColor)
        delta_e = delta_e_cie2000(bg_color, fg_color)
        while delta_e < 150 and delta_e > 250:  # 150-250
            # 创建画布背景色
            bg_b = np.random.randint(0, 255)
            bg_g = np.random.randint(0, 255)
            bg_r = np.random.randint(0, 255)
            # 文字前景色
            fg_b = np.random.randint(0, 255)
            fg_g = np.random.randint(0, 255)
            fg_r = np.random.randint(0, 255)
            # 计算前景和背景的彩色相似度
            bg_color = sRGBColor(bg_b, bg_g, bg_r)
            bg_color = convert_color(bg_color, LabColor)
            fg_color = sRGBColor(fg_b, fg_g, fg_r)
            fg_color = convert_color(fg_color, LabColor)
            delta_e = delta_e_cie2000(bg_color, fg_color)

        # 随机选择字体
        np.random.shuffle(self.font_name)
        cur_fonts = self.fonts.get(self.font_name[0])
        keys = list(cur_fonts.keys())
        np.random.shuffle(keys)
        font = cur_fonts.get(keys[0])
        text_size = font.getsize(text)

        # 根据字体大小创建画布
        img_w = text_size[0]
        img_h = text_size[1]

        # 文本区域上限
        h_space = np.random.randint(6, 20)
        w_space = 6
        h = img_h + h_space
        w = img_w + w_space
        canvas = Image.new('RGB', (w, h), (bg_b, bg_g, bg_r))
        draw = ImageDraw.Draw(canvas)

        # 随机平移
        start_x = np.random.randint(2, w_space-2)
        start_y = np.random.randint(2, h_space-2)

        # 绘制当前文本行
        draw.text((start_x, start_y), text, font=font, fill=(fg_b, fg_g, fg_r))
        img_array = np.array(canvas)
        # 透视失真
        src = np.float32([[start_x, start_y],
                          [start_x + w, start_y],
                          [start_x + w, start_y + h],
                          [start_x, start_y + h]])

        dst = np.float32([[start_x + np.random.randint(0, 10), start_y + np.random.randint(0, 5)],
                          [start_x + w - np.random.randint(0, 10), start_y + np.random.randint(0, 5)],
                          [start_x + w - np.random.randint(0, 10), start_y + h - np.random.randint(0, 5)],
                          [start_x + np.random.randint(0, 10), start_y + h - np.random.randint(0, 5)]])
        M = cv2.getPerspectiveTransform(src, dst)
        img_array = cv2.warpPerspective(img_array.copy(), M, (w, h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(bg_b, bg_g, bg_r))
        # Image.fromarray(img_array).show()
        # 随机旋转
        angle = np.random.randint(-8, 8)
        rotated = rotate_bound(img_array, angle=angle, bg_color=(bg_b, bg_g, bg_r))
        canvas = Image.fromarray(rotated)
        img_array = np.array(canvas.convert('CMYK'))[:,:,0:3]  # rgb to cmyk
        img_array = cv2.resize(img_array.copy(), (128, 32), interpolation=cv2.INTER_CUBIC)  # resize

        ndimg = Image.fromarray(img_array).convert('CMYK')
        # 保存
        save_path = os.path.join(self.save_path, '{}.jpeg'.format(i))  # 类别序列即文件名
        ndimg.save(save_path)

    def generator(self):
        n_samples = len(self.train_list)
        # 进度条
        widgets = ["数据集创建中: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=n_samples, widgets=widgets).start()

        for i in range(n_samples):
            # 绘制当前文本
            self.paint_text(self.train_list[i], i)
            pbar.update(i)

        pbar.finish()


if __name__ == '__main__':
    np.random.seed(0)  # 决定训练集的打乱情况
    DATASET_DIR = cfg.OUTPUT_DIR
    # 输出路径
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    psnr = []
    ssim = []
    # 实例化图像生成器
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    img_gen = TextGenerator(save_path=DATASET_DIR)
    img_gen.generator()