# coding=utf-8

# 语料集
corpus = 'config/sentences.txt'
dict = 'config/dict5990.txt'

# 字体文件路径
FONT_PATH = 'fonts/'

# 输出路径
OUTPUT_DIR = 'D:\Development\OCRdata\场景文本数据集/合成数据集'

# coding=utf-8
import os
# 路径参数
config_path = 'D:\Development\Workspace\TextImageGenerator\config'
corpus = os.path.join(config_path, 'sentences.txt')  # 语料集
dict = os.path.join(config_path, 'dict5990.txt')
FONT_PATH = 'fonts'  # 字体文件路径
n_samples = 32000
max_row_len = 10
max_label_len = 33

