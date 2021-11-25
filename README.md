# 前言

&nbsp;&nbsp;&nbsp;&nbsp;本文源码大部分是采用的[OpenCV实战（一）——简单的车牌识别](https://blog.csdn.net/weixin_41695564/article/details/79712393)这篇文章所提供的代码，对其代码进行了整合，追加了HSV、tesseract-OCR等内容。大佬文章中有对其步骤的详细讲解和分析，本文只是在原有基础上，进行了拓展和改造，细节内容可直接参考大佬的博文。由于大佬没有提供完整项目和模型，我这进行了自己简单的数据集构建和模型训练。

*ps：所有图片素材均源自网络，如果侵权可私信，立删。*

&nbsp;&nbsp;&nbsp;**开发环境：**
 - pycharm-2020
 - python-3.8.5
 - opencv-python-4.5.4.58
 - matplotlib-3.5.0
 - pip-21.2.3
 - Tesseract-OCR-5.0.0
 - numpy-1.21.4
 - sklearn-0.0.0
 - joblib-1.1.0
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/f0e14659329b43329e063d7629498d8c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBATG92ZeS4tuS8iuWNoea0m-aWrw==,size_18,color_FFFFFF,t_70,g_se,x_16)

 
## 工程下载
[码云](https://gitee.com/ikaros-521/python_opencv_LPR) [github](https://github.com/Ikaros-521/python_opencv_LPR)
 
# 效果图
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/941392c6654f4ecab32113292769ac76.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBATG92ZeS4tuS8iuWNoea0m-aWrw==,size_20,color_FFFFFF,t_70,g_se,x_16)
 ![请添加图片描述](https://img-blog.csdnimg.cn/fad159fc2d7946e186a369af5cb481fa.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBATG92ZeS4tuS8iuWNoea0m-aWrw==,size_20,color_FFFFFF,t_70,g_se,x_16)

# 简易流程图
![在这里插入图片描述](https://img-blog.csdnimg.cn/393f353018994d06b03cc4c79d2da219.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBATG92ZeS4tuS8iuWNoea0m-aWrw==,size_11,color_FFFFFF,t_70,g_se,x_16#pic_center)