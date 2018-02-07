### 本代码实现论文“DiabeticRetinopathyDetectionviaDeepConvolutionalNetworksfor DiscriminativeLocalizationandVisualExplanation”
#### 作者：梁新彦

### 运行过程

 1. 将训练数据放到data/train目录下，测试数据data/test目录下
 2. run preimage.py 将图片统一处理为512，512，可添加自己需要的图片预处理功能
 3. run run_memory.py 将数据一次导入到内存后训练模型，模型保存为model.h5
  (run.py从数据文件中读数据)
 4. run value2label.py 得到 kappa,accuracy,conf_matrix 三个指标

### To do：

更多数据预处理,合适的归一化方法

### Link

- [Keras数据增强](https://absentm.github.io/2016/06/14/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84Data-Augmentation%E6%96%B9%E6%B3%95%E5%92%8C%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0/)

注：可在run_memory.py中添加自己要进行的增强方式
