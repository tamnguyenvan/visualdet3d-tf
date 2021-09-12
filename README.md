# VisualDet3D Tensorflow
This repo aims to build a powerful framework for 2D & 3D visual detection. We are
inspired by [visualDet3D](https://github.com/Owen-Liuyuxuan/visualDet3D).

## Installation
Please install all packages in `requirements.txt`.

## Preparing dataset
Download [Kitti dataset](http://www.cvlibs.net/datasets/kitti/index.php) and extract it somewhere. Assuming you extracted it
to `PROJECT_DIR/kitti/`. The dataset structure should be as follows.
```
kitti/
|
|-- testing/
|   |
|   |-- calib/*.txt
|   |-- image_2/*.png
|   |-- image_3/*.png
|
|-- training/
|   |
|   |-- calib/*.txt
|   |-- image_2/*.png
|   |-- image_3/*.png
|   |-- label_2/*.txt
```
We have to precompute some metadata for training.
```
python tools/disparity_compute.py --config configs/yolostereo3d.yaml --data-dir kitti

python tools/imdb_precompute_3d.py --config configs/yolostereo3d.yaml --data-dir kitti

python tools/imdb_precompute_test.py --config configs/yolostereo3d.yaml --data-dir kitti/testing
```
If no errors raised, we would get a output directory named `preprocessed_output` as follows.
```
preprocessed_output/
|
|-- test/imdb.pkl
|
|-- training/
|   |
|   |--disp/*.png
|   |--anchor_mean_CLASS0.npy
|   |--anchor_mean_CLASS1.npy
|   |...
|   |--anchor_std_CLASS0.npy
|   |--anchor_std_CLASS1.npy
|   |...
|   |--imdb.pkl
|
|-- validation/imdb.pkl
```
## Training
Train Stereo3D model by this command. You might want to change settings and parameters in the config file.
```
python train.py --config configs/yolostereo3d.yaml
```