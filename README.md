# 클러스터링을 이용한 사진첩 정리

사진첩에서 문서류를 효과적으로 제거하는 프로젝트입니다. 얼굴, 음식, 풍경, 영수증 사진 각각 200개씩 총 800개의
사진 데이터를 이용해 데이터셋을 구성하였고 VGG16 모델로 feature를 추출하고 K-평균 알고리즘으로 군집화 한 뒤,
Python-tesseract로 군집에 포함된 문자열 길이가 제일 긴 군집을 제거함으로써 목표 성능을 달성하였습니다.

## Install

```bash
$ pip install numpy
$ pip install opencv-python
$ pip install pillow
$ pip install torchvision
$ pip install matplotlib
$ pip install tensorflow
$ pip install keras
$ pip install pytesseract

tesseract 설치: https://github.com/UB-Mannheim/tesseract/wiki
```

### Usage
```bash
1) 프로젝트에 필요한 라이브러리를 추가해줍니다.
```

```python
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
```

```bash
2) 이미지 데이터를 RGB채널의 224 * 224로 RESIZE 해줍니다.
```

```python
targerdir = r"./sample_data"
files = os.listdir(targerdir)

format = [".jpg"]
for (path, dirs, files) in os.walk(targerdir):
    for file in files:
        if file.endswith(tuple(format)):
            image = Image.open(path + "\\" + file)
            image = image.convert('RGB')
            image = image.resize((224,224))
            image.save("./sample_data/resize/" + file)
        else:
            print(path)
            print("InValid", file)

image_path = './sample_data/resize/'
img_list = os.listdir(image_path)
img_list_jpg = [img for img in img_list if img.endswith(".jpg")
```

```bash
3) VGG16 모델로 base_model을 선언합니다.
```

```python
base_model = VGG16(weights='imagenet')
```

```bash
4) 프로젝트에서는 VGG16 모델의 fc2 까지만을 이용할 것이기 때문에 출력층을 base_model의 fc2로 설정합니다.
```

```python
model = Model(inputs = base_model.input,outputs = base_model.get_layer('fc2').output)
```

```bash
5) 프로젝트 데이터에 대해 피쳐추출을 진행합니다.
```

```python
feature_map_list=[]
for i in img_list_jpg:
    img = Image.open(image_path + i)
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0],img.shape[1],img.shape[2]))
    img = preprocess_input(img)
    feature_map = model.predict(img)
    feature_map_list.append(feature_map[0])
```

```bash
6) PCA를 이용해 4096차원을 2차원으로 축소시킵니다.
```

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
feature_map_list_2 = pca.fit_transform(feature_map_list)
```

```bash
7) K-평균 알고리즘을 이용해 군집화시킵니다.
```

```python
from sklearn.cluster import KMeans
model = KMeans(init="k-means++", n_clusters=4).fit(X=feature_map_list_2.astype('double'))
```

```bash
8) 군집화 된 결과를 시각화 시켰습니다.
```

```python
plt.figure(figsize=(8, 4))
plot_decision_boundaries(model, feature_map_list_2)
plt.plot(mapx[:200], mapy[:200],'ro')
plt.plot(mapx[200:400], mapy[200:400],'go')
plt.plot(mapx[400:600], mapy[400:600],'bo')
plt.plot(mapx[600:], mapy[600:],'co')
plt.show()
```

```bash
9) Pytesseract library를 import하고 이를 이용해 가장 긴 문자열을 갖는 군집을 삭제목록으로 print 시켰습니다.
```

```python
from pytesseract import Output
import pytesseract
pytesseract.pytesseract.tesseract_cmd =r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img0=np.array(Image.open(image_path + img_list_jpg[sampleout(model.cluster_centers_[0],feature_map_list_2)]))
img1=np.array(Image.open(image_path + img_list_jpg[sampleout(model.cluster_centers_[1],feature_map_list_2)]))
img2=np.array(Image.open(image_path + img_list_jpg[sampleout(model.cluster_centers_[2],feature_map_list_2)]))
img3=np.array(Image.open(image_path + img_list_jpg[sampleout(model.cluster_centers_[3],feature_map_list_2)]))

text0 = pytesseract.image_to_string(img0)
text1 = pytesseract.image_to_string(img1)
text2 = pytesseract.image_to_string(img2)
text3 = pytesseract.image_to_string(img3)

start=0
if len(text0)>start:
    start=len(text0)
    ocr=0
if len(text1)>start:
    start=len(text1)
    ocr=1
if len(text2)>start:
    start=len(text2)
    ocr=2
if len(text3)>start:
    start=len(text3)
    ocr=3
print(ocr)

print('삭제목록',ocr,'번 cluster')
draw_sample_data(globals()[f'img_list_np{ocr}'])
```

#### Citations
```bash
@misc{}
```
