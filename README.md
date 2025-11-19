# LEAF-YOLO: Lightweight Edge-Real-Time Small Object Detection on Aerial Imagery

Implementation of paper - [LEAF-YOLO: Lightweight Edge-Real-Time Small Object Detection on
 Aerial Imagery](https://www.sciencedirect.com/science/article/pii/S2667305325000109)

<div align="center">
    <a href="./">
        <img src="./figure/params.png" width="40%"/>
    </a>
</div>

## Performance 

VisDrone2019-DET-val, test size = 640, using pytorch fp16 on 1xRTX 3090

| Model | #Param.(M) |FLOPs(G) |AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>S</sub><sup>val</sup> | AP<sub>M</sub><sup>val</sup> | batch 1 average time |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [**HIC-YOLOv5**](https://github.com/Jacoo-ai/HIC-Yolov5) | 9.3 | 30.9 | 24.9% | 43% | 15.8% | 35.2% | 29.7 ms |
| [**PDWT-YOLO**](https://github.com/1thinker1/PDWT-YOLO) | 6.44 | 24.5 | 24.3% | 42.6% | 15.9% | 33.4% | 22.2 ms |
| [**EdgeYOLO-T**](https://github.com/LSH9832/edgeyolo) | 5.5 | 27.24 | 21.8% | 38.5% | 12.4% | 32.5% | 29.93 ms |
| [**EdgeYOLO-S**](https://github.com/LSH9832/edgeyolo) | 9.3 | 45.32 | 23.5% | 40.8% | 13.8%| 34.8% | 38.18 ms |
| [**Drone-YOLO-N**](https://www.mdpi.com/2504-446X/7/8/526) | 3.05 | _ | 22.7% | 38.1% | _ | _ | _|
| [**Drone-YOLO-T**](https://www.mdpi.com/2504-446X/7/8/526) | 5.35 | _ | 25.6% | 42.8% | _ | _ | _ |
| [**Drone-YOLO-S**](https://www.mdpi.com/2504-446X/7/8/526) | 10.0 | _ | 27.0% | 44.3% | _ | _ | _ |
| [**LEAF-YOLO-N (Ours)**](https://github.com/highquanglity/LEAF-YOLO/blob/main/cfg/LEAF-YOLO/leaf-sizen/weights/best.pt) | **1.2** | **5.6** | **21.9%** | **39.7%** | **14.0%** | **30.6%** | **16.2** ms |
| [**LEAF-YOLO (Ours)**](https://github.com/highquanglity/LEAF-YOLO/blob/main/cfg/LEAF-YOLO/leaf-sizes/weights/best.pt) | **4.28** | **20.9** | **28.2%** | **48.3%** | **20.0%** | **38.0%** | **21.7 ms** |

## Installation

Conda environment (recommended)

``` shell
conda create -n leafyolo python=3.9
conda activate leafyolo
git clone https://github.com/highquanglity/LEAF-YOLO.git
cd LEAF-YOLO
pip install -r requirements.txt
```


## Testing

[`leafyolo-n.pt`](https://github.com/highquanglity/LEAF-YOLO/blob/main/cfg/LEAF-YOLO/leaf-sizen/weights/best.pt) [`leafyolo.pt`](https://github.com/highquanglity/LEAF-YOLO/blob/main/cfg/LEAF-YOLO/leaf-sizes/weights/best.pt)

``` shell
python test.py --data data/visdrone.yaml --img 640 --batch 16 --conf 0.01 --iou 0.5 --device 0 --weights cfg/LEAF-YOLO/leaf-sizes/weights/best.pt --name test --no-trace --save-json
```
## Training

Data preparation

* Download VisDrone2019-DET dataset images ([train](https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view), [val](https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view), [test](https://drive.google.com/file/d/1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V/view)) or follow this document of Ultralytics [Visdrone2019-DET](https://docs.ultralytics.com/datasets/detect/visdrone/). If you have previously used a different version of YOLO, we strongly recommend that you delete `train.cache` and `val.cache` files.
Single GPU training

``` shell
python train.py --workers 16 --device 0 --batch-size 16 --epochs 1000 --data data/visdrone.yaml --img 640 640 --cfg cfg/LEAF-YOLO/leaf-sizes.yaml --weights '' --hyp data/hyp.scratch.visdrone.yaml --cache --name leafyolo
```

Multiple GPU training

``` shell
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/visdrone.yaml --img 640 640 --cfg cfg/LEAF-YOLO/leaf-sizen.yaml --weights '' --name leafyolo_n --hyp data/hyp.scratch.visdrone.yaml
```

## Visualisation

See [grad_cam_visualize.ipynb](grad_cam_visualize.ipynb)

<div align="center">
    <a href="./">
        <img src="figure/heatmap_sizen.png" width="75%"/>
    </a>
</div>

## Inference

On video:
``` shell
python detect.py --weights cfg/LEAF-YOLO/leaf-sizes/weights/best.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
```

On image:
``` shell
python detect.py --weights cfg/LEAF-YOLO/leaf-sizes/weights/best.pt --conf 0.25 --img-size 640 --source yourimage.png
```

## Export (Follow YOLOv7)
**Pytorch to ONNX with NMS (and inference)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7onnx.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
```shell
python export.py --weights cfg/LEAF-YOLO/leaf-sizes/weights/best.pt --grid --end2end --simplify \
        --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```

**Pytorch to TensorRT with NMS (and inference)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7trt.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

```shell
python export.py --weights  cfg/LEAF-YOLO/leaf-sizes/weights/best.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o best.pt.onnx -e leafs.trt -p fp16
```

**Pytorch to TensorRT another way** <a href="https://colab.research.google.com/gist/AlexeyAB/fcb47ae544cf284eb24d8ad8e880d45c/yolov7trtlinaom.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <details><summary> <b>Expand</b> </summary>


```shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
python export.py --weights yolov7-tiny.pt --grid --include-nms
git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o yolov7-tiny.onnx -e yolov7-tiny-nms.trt -p fp16

# Or use trtexec to convert ONNX to TensorRT engine
/usr/src/tensorrt/bin/trtexec --onnx=yolov7-tiny.onnx --saveEngine=yolov7-tiny-nms.trt --fp16
```

</details>

Tested with: Python 3.7.13, Pytorch 1.12.0+cu113

## Real-time on Jetson AGX Xavier with TensorRT

Test with models converted e2e, including EfficientNMS, to fp16 format using TensorRT8.5.2 on Jetson AGX Xavier. 

| Model |  AP<sub>50</sub><sup>val</sup> | FPS<sub>b=1</sub><sup>fp16.trt</sup> |FPS<sub>b=16</sub><sup>fp16.trt</sup> |
| :-- | :-: | :-: | :-: |
| EdgeYOLO-T | 38.5% | 27 | 65 |
| EdgeYOLO-S | 40.8% | 21 | 54 |
| **LEAF-YOLO-N** | **39.7%** | **56** | **147** |
| **LEAF-YOLO** | **48.3%** | **32** | **87** |


## Citation

```
@article{NGHIEM2025200484,
title = {LEAF-YOLO: Lightweight Edge-Real-Time Small Object Detection on Aerial Imagery},
journal = {Intelligent Systems with Applications},
volume = {25},
pages = {200484},
year = {2025},
issn = {2667-3053},
doi = {https://doi.org/10.1016/j.iswa.2025.200484},
url = {https://www.sciencedirect.com/science/article/pii/S2667305325000109},
author = {Van Quang Nghiem and Huy Hoang Nguyen and Minh Son Hoang},
keywords = {Aerial imagery, UAV imagery, Small object detection, Edge-real-time algorithm, You only look once (YOLO)},
abstract = {Advances in Unmanned Aerial Vehicles (UAVs) and deep learning have spotlighted the challenges of detecting small objects in UAV imagery, where limited computational resources complicate deployment on edge devices. While many high-accuracy deep learning solutions have been developed, their large parameter sizes hinder deployment on edge devices where low latency and efficient resource use are essential. To address this, we propose LEAF-YOLO, a lightweight and efficient object detection algorithm with two versions: LEAF-YOLO (standard) and LEAF-YOLO-N (nano). Using Lightweight-Efficient Aggregating Fusion along with other blocks and techniques, LEAF-YOLO enhances multiscale feature extraction while reducing complexity, targeting small object detection in dense and varied backgrounds. Experimental results show that both LEAF-YOLO and LEAF-YOLO-N outperform models with fewer than 20 million parameters in accuracy and efficiency on the Visdrone2019-DET-val dataset, running in real-time (>30 FPS) on the Jetson AGX Xavier. LEAF-YOLO-N achieves 21.9% AP.50:.95 and 39.7% AP.50 with only 1.2M parameters. LEAF-YOLO achieves 28.2% AP.50:.95 and 48.3% AP.50 with 4.28M parameters. Furthermore, LEAF-YOLO attains 23% AP.50 on the TinyPerson dataset, outperforming models with ≥ 20 million parameters, making it suitable for UAV-based human detection.}
}
```

## Acknowledgements

We would like to extend our gratitude to WongKinKyu for their work on YOLOv7 and Ultralytics for YOLOv5. This project builds upon the foundations laid by these incredible frameworks, and their contributions have been invaluable to the development of this repository.

* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
