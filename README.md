# Object_detection_estimated_sclales
This repository contains the public release of the python implementation of Detection of [Imaged Objects with Estimated Scales](http://www.scitepress.org/PublicationsDetail.aspx?ID=Jr1S7apgu68=&t=1),

If you find this code useful, please cite our paper:
```
@conference{visapp19,
author={Xuesong Li. and Ngaiming Kwok. and Jose E. Guivant. and Karan Narula. and Ruowei Li. and Hongkun Wu.},
title={Detection of Imaged Objects with Estimated Scales},
booktitle={Proceedings of the 14th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 5: VISAPP,},
year={2019},
pages={39-47},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0007353600390047},
isbn={978-989-758-354-4},
}
```

## Installation
Implemented and tested on Ubuntu 16.04 with Python 3.5 and Tensorflow 1.8.0.

1. Clone this repo
    ```
    >>git clone https://github.com/Benzlxs/Object_detection_estimated_sclales --recurse-submodules
    ```
2. Install [tensorflow-1.8.0](https://www.tensorflow.org/install/)

3. Install Python dependencies
    ```
    >>cd Object_detection_estimated_sclales
    >>pip3 install -r requirements.txt
    ```

4. Compile integral image library in wavedata
    ```
    >>sh scripts/install/build_integral_image_lib.bash
    ```

5. Protobufs are used to configure model and training parameters. Before the framework can be used, the protos must be compiled:
    ```
    >>sh odes/protos/run_protoc.sh
    ```


## Preparation
1. Download [Kitti Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d), and place it in your home folder at `~/Kitti/object`

2. Split data (split_ratio = 0.75 in our setting)
    ```
    >>python odes/utils/split_dataset.py
    ```
    The `train.txt` and `val.txt` will be generated, and copy them into `~/Kitti/object`. If you want to try other split   ratio, adjust split_ratio in split_dataset.py

3. Download [planes](https://drive.google.com/drive/folders/1c5z3NqoLw78NvGWoF_3MBnIsyRI41xSP?usp=sharing) into `~/Kitti/object`

4. Download the pretrained model [VGG16](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) in tensorflow models, and unzip the pretained model into `odes/data/pretrained_models`

### Training
1. Generate preprocessed data including estimated scale size from depth.
    ```
    >>chmod +x generate_img_batch.sh
    >>./generate_img_batch.sh
    ```
    The configuration file can be found under directory `./odes/configs/mb_preprocessing/`. Changing parameter settings like area size, density threshold etc. in the config file according to your application.

2. Start to train.
    ```
    >>chmod +x train.sh
    >>./train.sh
    ```
    The training configuration can be found under directory `./odes/configs/` where you can find the max number of iterations, learning rate, optimizer and pretrained model dir etc.
    
### Evaluating
1. Run evaluation code
    ```
    >>chmod +x evaluate.sh
    >>./evaluate.sh
    ```
    If you have the multiple GPU, you can run trianing and evluating simultaneously on different GUPs by setting `CUDA_VISIBLE_DEVICES` in the evaluate.sh file.
   
2. Evaluting results.

    | AP       | Easy   |Moderate|  Hard  |
    |:--------:|:------:|:------:|:------:|
    |  Car     |93.53   |90.89   | 90.62  |
    |Pedestrain| 80.39  |  74.85 |66.76   |
    | Cyclist  | 81.33  | 76.73  |   71.47|
    
    Evaluate your model around 380000 Iterations, you should be able to achieve above accuracy

### Acknowledge
Thanks to the team of Jason Ku , we have benifited a lot from their previous work [AVOD (Aggregate View Object Detection)](https://arxiv.org/abs/1712.02294) and his [code](https://github.com/kujason/avod).
