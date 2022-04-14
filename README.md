# MosaicMVS
Official source code of paper MosaicMVS: Mosaic-based Omnidirectional Multi-view Stereo for Indoor Scenes

As we cannot share the entire dataset, only provide some pictures of scene1 for testing.

For testing our code, run the shell file:

# Installation requirements
We run this code using GeForce RTX 3090 in following environment.

```
python >=3.7
Pytorch ==1.10
CUDA >=9.0
```

# Testing
To run this code,
```
git clone https://github.com/min-jung-shin/MosaicMVS.git
sh test_mosaic.sh
```

# Fusion
To fusion output depthmaps
```
CUDA_VISIBLE_DEVICES=0,1 python fusioncas.py -data <dir_of_depths> --pair <dir_of_pair> --vthresh 3 --pthresh .8,.8,.8 --outmask <dir_of_masks> --out_dir <dir_of_output pointcloud>
```
# Acknowledgements
Thanks to Xiaodong Gu for opening source of his excellent work CasMVSNet. Thanks to Jingyang Zhang of his excellent work Vis-MVSNet.
