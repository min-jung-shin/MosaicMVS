# MosaicMVS: Mosaic-based Omnidirectional Multi-view Stereo for Indoor Scenes
<img width="1835" alt="Figure_1_2" src="https://user-images.githubusercontent.com/65907536/163343364-63e6b2ca-7ff4-47ac-8b65-05ebb0cf4297.png">

Official source code of paper MosaicMVS: Mosaic-based Omnidirectional Multi-view Stereo for Indoor Scenes

As we cannot share the entire dataset, only provide some pictures of scene1 for testing.

For testing our code, run the shell file:

## Installation requirements 

We run this code using GeForce RTX 3090 in following environment.

```
python >=3.7
Pytorch ==1.10
CUDA >=9.0
```

## Dataset

Parts of the scene1 images in SAOI dataset are uploaded in scene1 folder for testing.
For more datasets you needed, please contact us.

---
## Testing

To run this code,
```
git clone https://github.com/min-jung-shin/MosaicMVS.git
sh test_mosaic.sh
```

## Fusion

To fusion output depthmaps,

```
CUDA_VISIBLE_DEVICES=0,1 python fusioncas.py -data <dir_of_depths> --pair <dir_of_pair> --vthresh 3 --pthresh .8,.8,.8 --outmask <dir_of_masks> --out_dir <dir_of_output pointcloud>
```
For reconstrction evaluation, refer to the code in the python fusioncas.py. 

```
("total.mean: ", sum(total)/len(total))
```
---

## Evaluation of estimated depth map

To evaluate depth map, COLMAP sparse reconstruction depthmaps are needed.
You can run the [COLMAP](https://github.com/colmap/colmap). 




## Customed mosaic images

If you capture customed mosaic images, you must run the [COLMAP](https://github.com/colmap/colmap) to obtain undistorted image, and camera parameters.




## Acknowledgements

Thanks to Xiaodong Gu for opening source of his excellent work CasMVSNet. Thanks to Jingyang Zhang of his excellent work Vis-MVSNet.
