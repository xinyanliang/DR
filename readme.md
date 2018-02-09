### 本代码实现论文“DiabeticRetinopathyDetectionviaDeepConvolutionalNetworksfor DiscriminativeLocalizationandVisualExplanation”
#### 作者：梁新彦

### 训练过程

1. 将训练数据放到data/train目录下，测试数据data/test目录下
2. run preimage.py 将图片统一处理为512，512，可添加自己需要的图片预处理功能
 ```
 python preimage.py
 ```
3. run_memory128.py的训练权重，用作run_memory256.py的初始权重。run_memory256.py的训练好的权重用作run_memory512.py的初始权重。
 依次运行下面的命令。注：这三个运行顺序，必须按如下所示，且必须一个执行完后，在运行下一个。
 ```
 python run_memory128.py
 ```
 ```
 python run_memory256.py
 ```
 ```
 python run_memory512.py
 ```
  
4. run value2label.py 得到 kappa,accuracy,conf_matrix 三个指标
 ```
 python value2label.py
 ```

### To do：

- [x] 将原始图片处理三种尺度(128,256,512)的图像
- [ ] 更多数据预处理,合适的归一化方法

### Q&A

1. __Q__:加入数据增强后，训练速度变慢。表现为，通过nvidia-smi -l 观察gpu使用情况，发现gpu使用率变化幅度太大，有时光谱使用率居然为0 哎...


__A__: fit_generator中，默认不适用多进程，且只是用一个cpu core。我们只需自己手动设置这两个参数即可。

```
model.fit_generator(
    use_multiprocessing=True, #默认False
    workers=cpus) #默认1
```

### Link

- [Keras数据增强](https://absentm.github.io/2016/06/14/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84Data-Augmentation%E6%96%B9%E6%B3%95%E5%92%8C%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0/)
- [python实现的数据增强](https://github.com/aleju/imgaug)

