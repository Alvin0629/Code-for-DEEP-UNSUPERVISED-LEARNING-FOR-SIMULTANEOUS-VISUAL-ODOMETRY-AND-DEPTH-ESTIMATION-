# Code for 2019 ICIP paper


This code implemented the method described in the following paper:

## Deep Unsupervised Learning for Simultaneous Visual Odometry and Depth Estimation.

#### 2019 IEEE International Conference on Image Processing (ICIP)



To train the network, use:
```
python3 train.py sfm-learner/KITTI_RAW_DATA/ -b4 -m0.2 -s0.1 --epochs 500 --sequence-length 3 --log-output
```

#sfm-learner/KITTI_RAW_DATA/# is the path to save the dataset. 


To infer the network, use:
```
python3 run_inference.py --pretrained pretrained_model/Dispnet --dataset-dir test_dir/ --output-dir output_dir/
```

If you summarize relevant works or refer to the code, please cite:
```
@inproceedings{lu2019deep,
  title={Deep Unsupervised Learning for Simultaneous Visual Odometry and Depth Estimation},
  author={Lu, Yawen and Lu, Guoyu},
  booktitle={2019 IEEE International Conference on Image Processing (ICIP)},
  pages={2571--2575},
  year={2019},
  organization={IEEE}
}
```


This implementation is borrowed from SfMLearner paper.
