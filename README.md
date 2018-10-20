# Object_detection_estimated_sclales

The paper is still underview, and will be released here after it is accepted.

## Getting Started
Implemented and tested on Ubuntu 16.04 with Python 3.5 and Tensorflow 1.8.0.

1. Clone this repo
```bash
git clone https://github.com/Benzlxs/Object_detection_estimated_sclales --recurse-submodules
```
2. Install [tensorflow-1.8.0](https://www.tensorflow.org/install/)

3. Install Python dependencies
```bash
cd Object_detection_estimated_sclales
pip3 install -r requirements.txt
```

4. Compile integral image library in wavedata
```bash
sh scripts/install/build_integral_image_lib.bash
```

5. Protobufs are used to configure model and training parameters. Before the framework can be used, the protos must be compiled:
```bash
sh odes/protos/run_protoc.sh
```


## Training
### [Kitti Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):

1. Download the data and place it in your home folder at `~/Kitti/object`

2. Split data (split_ratio = 0.75 in our setting)
```
python odes/utils/split_dataset.py
```
The `train.txt` and `val.txt` will be generated, and copy them into `~/Kitti/object`. If you want to try other split ratio, adjust split_ratio in split_dataset.py

3. Download [planes](https://drive.google.com/drive/folders/1c5z3NqoLw78NvGWoF_3MBnIsyRI41xSP?usp=sharing) into `~/Kitti/object`

### Mini-batch Generation
The training data needs to be pre-processed to generate mini-batches. To configure the mini-batches, you can modify `odes/configs/mb_preprocessing/rpn_[class].config`. You also need to select the *class* you want to train on. Inside the `scripts/preprocessing/gen_img_mini_batches.py` select the classes to process. By default it processes the *Car* and *People* classes, where the flag `process_[class]` is set to True. The People class includes both Pedestrian and Cyclists. *pdestrain* and *cyclist* can also be processed separately.

Note: This script does parallel processing with `num_[class]_children` processes for faster processing. This can also be disabled inside the script by setting `in_parallel` to `False`.

```bash
cd avod
python scripts/preprocessing/gen_mini_batches.py
```

Once this script is done, you should now have the following folders inside `avod/data`:
```
data
    label_clusters
    mini_batches
```

### Acknowledge
Thanks to the team of Jason Ku , we have benifited a lot from their previous work [AVOD (Aggregate View Object Detection)](https://arxiv.org/abs/1712.02294) and his [code](https://github.com/kujason/avod).
