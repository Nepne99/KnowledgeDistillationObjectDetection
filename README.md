# Knowledge Distillation for Object Detection

해당 프로젝트는 Object Detection에서 Knowledge Distillation 경량화 방법을 사용하여 Student 모델의 성능을 향상시킵니다.
Faster R-CNN 모델 구조와 ResNet backbone을 사용하며, Pascal VOC 데이터셋을 기반으로 실험을 진행했습니다.

## 구성
+ 하드웨어
  + **CUDAToolKit 10.1을 지원하는 NVIDIA Turing Architecture 이하 && Keppler Architecture 이상의 그래픽 카드**
  + ![CUDAToolKitVersion](https://github.com/user-attachments/assets/235eb9e8-6e4a-42a6-b774-7f8049d59f44)
+ OS & SW
  + Ubuntu 22.04
  + Nvidia docker
 
## 도커 컨데이너 및 데이터셋 준비
+ Docker Image tar File 받기\
  https://drive.google.com/file/d/1_AB-jfFZrWaKapZ6YJVECombbGnMsKKG/view?usp=sharing
+ Docker tar File -> Docker Image\
  ```docker load -i defeat.tar```
+ Docker Image -> Docker Container\
  ```docker run -it --gpus all --name defeat --shm-size=32G -v ./mmdetection200_defeat:/mmdetection /bin/bash```
+ 데이터셋은 ```/mmdetection200_defeat/data/VOCdevkit``` 위치
+ VOC 데이터셋은 2007, 2012로 구성 (별도 다운로드)
## 실행
+ Train
  + Teacher Model Train\
    Dist train: ```python -m torch.distributed.launch --nproc_per_node=2 tools/train.py configs/faster_rcnn/voc_faster_rcnn_r101.py --launcher pytorch --work-dir ./work/voc/r101 --validate```\
    Single train: ```python tools/train.py configs/faster_rcnn/voc_faster_rcnn_r101.py --work-dir ./work/voc/r101 --validate```
  + Baseline Student Model Train\
  + Dist train: ```python -m torch.distributed.launch --nproc_per_node=2 tools/train.py configs/faster_rcnn/voc_faster_rcnn_r50.py --launcher pytorch --work-dir ./work/voc/r50 --validate```\
    Single train: ```python tools/train.py configs/faster_rcnn/voc_faster_rcnn_r50.py --work-dir ./work/voc/r50 --validate```
  + Distilling Teacher to Student Train\
    Dist train: ```python -m torch.distributed.launch --nproc_per_node=2 tools/train_kd.py --gpus 2 --launcher pytorch --validate --work-dir work/kd/r101-50-DeFeat/ --config configs/kd_faster_rcnn/voc_stu_faster_rcnn_r50_decouple_neck.py --config-t configs/kd_faster_rcnn/voc_tea_faster_rcnn_r101.py --checkpoint-t work/voc/r101/latest.pth```
+ Test
  + Teacher Model Test\
    ```python tools/test.py configs/faster_rcnn/voc_faster_rcnn_r101.py /mmdetection/work/voc/r101/latest.pth --eval mAP```
  + Baseline Student Model Test\
    ```python tools/test.py configs/faster_rcnn/voc_faster_rcnn_r50.py /mmdetection/work/voc/r50/latest.pth --eval mAP```
  + Distilled Student Test\
    ```python tools/test.py configs/faster_rcnn/voc_faster_rcnn_r50.py /mmdetection/work/kd/r101-50-DeFeat/latest.pth --eval mAP```
