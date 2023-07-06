# 多模态情感分析
图像+文本数据的双模态情感分析分类问题

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:

- torch==1.10.1

- sklearn==1.0.2

- transformers==4.30.2

You can simply run 

```python
pip install -r requirements.txt
```

## Repository structure
```python
|-- data/ # 图片数据和文本数据
|-- main.py # the main code
|-- main.ipynb# main.py的ipynb形式
|-- predict.txt  #test_without_label.txt的预测文件
|-- train.txt # 训练集
|-- txt_total.txt# data/下所有文本文件内容的汇总
|-- test_without_label.txt  #测试集
|-- requirements.txt 
|-- README.md
|-- img/ #10205501409_陈沁文_实验五.md的图片
|-- 10205501409_陈沁文_实验五.md #报告
```

## Run 
option=0为只有图像数据输入，option=1为只有文本数据输入，option=2为图像+文本输入的双模态融合模型，默认为2
```python
python main.py
```
或
```python
python main.py --option 2
```

