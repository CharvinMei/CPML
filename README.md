# CPML: Category Probability Mask Learning for Fine-Grained Visual Classification
<div align="center">
  <img src="https://github.com/CharvinMei/CPML/blob/main/pictures/Fig1.jpg">
</div>

**Installation**

Make sure you have **Python>=3.6** installed on your machine.
Install python dependencies by running:
`pip install -r requirements.txt`

________________________________

**Datasets:**

Place the public data sets correspondingly in the following directories：

`./datasets/CUB-200-2011`

`./datasets/FGVC_Aircraft1`

`./datasets/Stanford_cars`
________________________________
**Traing:**

_**For CUB-200-2011 dataset:**_
`python train_CUB_resnet50.py`
or
`python train_CUB_mobilenetv3`
or
`python train_CUB_densenet161`

**_For FGVC_Aircraft dataset:_**
`python train_Air_resnet50.py`
or
`python train_Air_mobilenetv3`
or
`python train_Air_densenet161`

**_For Stanford_cars dataset:_**
`python train_Car_resnet50.py`
or
`python train_Car_mobilenetv3`
or
`python train_Car_densenet161`
