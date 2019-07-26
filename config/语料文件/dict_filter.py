"coding = utf-8"
# 删除语料中的生僻字
import codecs


dictfile = "dict4200.txt"
corpus_file = "renri.txt"
output = "renri_clean.txt"
dict = []
with codecs.open(dictfile, mode='r', encoding='utf-8') as f:
    # 按行读取语料
    for line in f:
        # 当前行单词去除结尾，为了正常读取空格，第一行两个空格
        word = line.strip('\r\n')
        # 只要没超出上限就继续添加单词
        dict.append(word)

with codecs.open(corpus_file, mode='r', encoding='utf-8') as f:
    with codecs.open(output, mode='w', encoding='utf-8') as output:
        for line in f:
            for each in line:
                # 去除生僻字
                try:
                    dict.index(each)
                    output.write(each + '\n')
                except:
                    print("字典不包含{}, 忽略".format(each))