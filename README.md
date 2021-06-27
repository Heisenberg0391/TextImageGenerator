# TextImageGenerator
该脚本根据语料文件生成对应的图像文件，适用于文本识别等计算机视觉任务。
>new feature: 支持多线程，大幅提升脚本速度
>
根目录下的fonts文件夹用于存放字体文件, imageset文件夹用于存放输出图像和映射表, 
config/cfg.py中设置相关参数并存放语料文件:
>(1)dict.txt是字典，字典文件应保证每行一个字符
>(2)sentences.txt是语料集
>(3)OUTPUT_DIR是文本图片和映射文件的输出路径
>(4)n_samples用于控制输出的图片总数

>
脚本运行：python sample_generator.py
>
![效果图1](/imageset/0.jpeg)
>
![效果图2](/imageset/1.jpeg)
>
![效果图1](/imageset/6.jpeg)
>
![效果图2](/imageset/7.jpeg)
>
![效果图2](/imageset/9.jpeg)
>
文件-类别序列映射表
>
![效果图3](/mapping.png)
>
语料集:
>
![效果图3](/sentences.png)


