# -*- coding:utf8 -*-
# 生成合成文本图像用于训练
import codecs
import numpy as np
import config.cfg as cfg
import os
from PIL import Image, ImageDraw, ImageFont
import glob
import progressbar
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.style.use("ggplot")


class TextGenerator():
    def __init__(self, img_w, img_h):
        """初始化参数来自config文件
        """
        super(TextGenerator, self).__init__()
        self.img_w = img_w
        self.img_h = img_h
        self.img_lim = cfg.canvas_lim  # 图片最大宽度
        # 语料参数
        self.dictfile = cfg.dict  # 字典
        self.dict = []
        self.corpus_file = cfg.corpus  # 语料集
        # 加载字体文件
        self.font_factor = 1  # 控制字体大小
        # 选择字体
        root = tk.Tk()
        root.withdraw()
        self.font_path = filedialog.askopenfilename()
        self.load_fonts(self.font_factor, self.font_path)
        # 加载语料集
        self.build_dict()
        self.build_train_list(cfg.n_samples, cfg.sentence_lim)

    def load_fonts(self, factor, font_path):
        """ 加载字体文件并设定字体大小
        """
        self.fonts = []
        # 字体完整路径
        # font_path = os.path.join(cfg.FONT_PATH, "*.ttf")
        # 获取全部字体路径，存成list
        fonts = list(glob.glob(font_path))
        # 遍历字体文件
        for each in fonts:
            # 调整字体大小
            font = ImageFont.truetype(each, int(self.img_h*factor), 0)
            self.fonts.append(font)

    def build_dict(self):
        """ 打开字典，加载全部字符到list
            每行是一个字
        """
        with codecs.open(self.dictfile, mode='r', encoding='utf-8') as f:
            # 按行读取语料
            for line in f:
                # 当前行单词去除结尾，为了正常读取空格，第一行两个空格
                word = line.strip('\r\n')
                # 只要没超出上限就继续添加单词
                self.dict.append(word)
        # 最后一位作为空白符
        self.blank_label = len(self.dict)

    def mapping_list(self):
        # 写图像文件名和类别序列的对照表
        file_path = os.path.join(cfg.OUTPUT_DIR, 'map_list.txt')
        with codecs.open(file_path, mode='w', encoding='utf-8') as f:
            for i in range(len(self.train_list)):
                f.write("{}.png {} \n".format(i, self.train_list[i]))

    def build_train_list(self, num_rows, max_row_len=None):
        # 过滤语料，留下适合的内容组成训练list
        assert max_row_len <= self.img_lim
        self.num_rows = num_rows
        self.max_row_len = max_row_len
        sentence_list = []
        self.train_list = []

        with codecs.open(self.corpus_file, mode='r', encoding='utf-8') as f:
            # 按行读取语料
            for line in f:
                sentence = line.rstrip().replace(' ', '')  # 当前行单词
                if len(sentence) <= max_row_len and len(sentence_list) < num_rows:
                    # 只要句子长度不超过画布上限且句子数量没超出上限就继续添加
                    sentence_list.append(sentence)
                elif len(sentence) > max_row_len and len(sentence_list) < num_rows:
                    # 截断句子
                    sentence_list.append(sentence[0:max_row_len])

        if len(sentence_list) < self.num_rows:
            raise IOError('Could not pull enough words corpus file.')

        for i, sentence in enumerate(sentence_list):
            # 遍历语料中的每一句(行)
            # 将单词分成字符，然后找到每个字符对应的整数ID list
            label_sequence = []
            for j, word in enumerate(sentence):  # 检查句子中是否包含生僻字
                try:
                    index = self.dict.index(word)
                    label_sequence.append(index)
                except ValueError:
                    print("字典不包含：{}，已忽略".format(word))
                    sentence_list[i] = sentence_list[i][0:j] + sentence_list[i][j+1:]  # 从该句中删除生僻字

        self.train_list = sentence_list  # 过滤后的训练集
        np.random.shuffle(self.train_list)  # 打乱顺序
        self.mapping_list()  # 保存图片名和类别序列的 map list

    def paint_text(self, text, i):
        """ 使用PIL绘制文本图像，传入画布尺寸，返回文本图像
        :param h: 画布高度
        :param w: 画布宽度
        """
        # 创建画布
        canvas = np.zeros(shape=(self.img_h, self.img_w), dtype=np.uint8)
        canvas[0:] = 255
        # 转换图像模式，保证合成的两张图尺寸模式一致
        ndimg = Image.fromarray(canvas).convert('RGBA')
        draw = ImageDraw.Draw(ndimg)

        font = self.fonts[-1]
        text_size = font.getsize(text)  # 获取当前字体下的文本区域大小

        # 自动调整字体大小避免超出边界, 至少留白水平20%
        margin = [self.img_w - int(0.2*self.img_w), self.img_h - int(0.2*self.img_h)]
        while (text_size[0] > margin[0]) or (text_size[1] > margin[1]):
            self.font_factor -= 0.1
            self.load_fonts(self.font_factor, self.font_path)
            font = self.fonts[-1]
            text_size = font.getsize(text)

        # 随机平移
        horizontal_space = self.img_w - text_size[0]
        vertical_space = self.img_h - text_size[1]
        start_x = np.random.randint(2, horizontal_space-2)
        start_y = np.random.randint(2, vertical_space-2)

        # 绘制当前文本行
        draw.text((start_x, start_y), text, font=font, fill=(0, 0, 0, 255))

        # 画图看一下
        # img_array = np.array(ndimg)
        # plt.figure(1)
        # plt.imshow(img_array)
        # plt.show()
        save_path = os.path.join(cfg.OUTPUT_DIR, '{}.png'.format(i))  # 类别序列即文件名
        ndimg.save(save_path)

    def text_generator(self):
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
    # 输出路径
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    # 参数
    img_w = 300
    img_h = 40

    # 实例化图像生成器
    img_gen = TextGenerator(img_w=img_w, img_h=img_h)
    img_gen.text_generator()
