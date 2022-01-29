# Grocery objects detection and classification

<img src="big_picture.png" >

### Setting up the Environment
0. Create conda environment.
```bash
conda create -n tensorflow pip python=3.9 # call it tensorflow
conda activate tensorflow # activate it.
```
1. Install tensorflow
```bash
pip install --ignore-installed --upgrade tensorflow==2.5.0
# test installation
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```
Seeing the below message means that python is compiled with CUDA.
```bash
2021-12-18 14:07:37.042797: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1418] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 13803 MB memory) -> physical GPU (device: 0, name: Tesla T4, pci bus id: 0000:00:1e.0, compute capability: 7.5)
tf.Tensor(333.1103, shape=(), dtype=float32)
```
2. Clone the official TF detection models.
```bash
git clone https://github.com/tensorflow/models.git
```
3. Install protobuf for tf.
```bash
cd models/research
protoc object_detection/protos/*.proto --python_out=.
```
4. Build Python Cocoapi tools and import into research directory.
```bash
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make # if errors out, then pip install cython
cp -r pycocotools ../../ # models/research
```
5. Install Object Detection API
```bash
cd ../../ # models/research
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```
6. Run tests to confirm proper installation.
```bash
python object_detection/builders/model_builder_tf2_test.py
```
### Creating folder structure for custom dataset training.

1. Create the directories.
```bash
cd ../../ # back to /
mkdir training_demo
cd training_demo
mkdir annotations models pre-trained-models images exported-models
```
2. Download a pretrained Tensorflow model into pre-trained-models directory. Tensorflow model zoo - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
```bash
cd .. # back to /
cd training_demo/pre-trained-models
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz
tar xvf ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz
rm -r ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz
```
3. Add the images, labels under train/test in training_demo/images. Create labels and annotations for the images using labelImg tool which can be installed via pip.
4. Create label.pbtxt file under training_demo annotations with below class information.
```yaml
item {
    id: 1
    name: 'apple_kashmir'
}

item {
    id: 2
    name: 'banana_red'
}
```
5. Generate tfrecord files for train/test sets. For that grab code from https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html and save into ```generate_tfrecord.py``` file.
```bash
# train
python generate_tfrecord.py -i training_demo/images/train -x training_demo/images/train -l training_demo/annotations/label_map.pbtxt -o training_demo/annotations/train.record

# test
python generate_tfrecord.py -i training_demo/images/test -x training_demo/images/test -l training_demo/annotations/label_map.pbtxt -o training_demo/annotations/test.record
```

**Side Note:** Command to copy annotations from local mac to aws GPU 
```bash
scp -r -i hari_cisco.pem training_demo/annotations ubuntu@ec2-65-0-119-51.ap-south-1.compute.amazonaws.com:/home/ubuntu/tf-grocery-object-detection/training_demo
```
6. Configuring training job. Lets get the pipeline.config under the pretrained directory right into our custom model training dir.
```bash
mkdir models/my_ssd_resnet50_v1_fpn
cp pre-trained-models/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8/pipeline.config models/my_ssd_resnet50_v1_fpn/
```
Apply edits as below.
```yaml
num_classes: 2 # Set this to the number of different label classes
batch_size: 8 # Increase/Decrease this value depending on the available memory (Higher values require more memory and vice-versa)
fine_tune_checkpoint: "pre-trained-models/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0" # Path to checkpoint of pre-trained model
fine_tune_checkpoint_type: "detection" # Set this to "detection" since we want to be training the full detection model
use_bfloat16: false # Set this to false if you are not training on a TPU
label_map_path: "annotations/label_map.pbtxt" # Path to label map file
input_path: "annotations/train.record" # Path to training TFRecord file
metrics_set: "coco_detection_metrics"
use_moving_averages: false
label_map_path: "annotations/label_map.pbtxt" # Path to label map file
input_path: "annotations/test.record" # Path to testing TFRecord
```
7. Train it. Copy training script right into the training_demo dir.
```bash
cp models/research/object_detection/model_main_tf2.py training_demo
# execute training command from training_demo
python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config

# for centernet
python model_main_tf2.py --model_dir=models/my_ssd_mobilenet_v2_320x320_coco17_tpu-8 --pipeline_config_path=models/my_ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config
```

## Exporting trained model
```bash
# execute from /. copy the exporter_main script into training_demo.
cp models/research/object_detection/exporter_main_v2.py training_demo/
cd training_demo
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path models/my_ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config --trained_checkpoint_dir models/my_ssd_mobilenet_v2_320x320_coco17_tpu-8 --output_directory exported-models/my_model
# copy this back into mac. from / in local run.
scp -r -i "hari_cisco.pem" -r ubuntu@ec2-3-111-49-20.ap-south-1.compute.amazonaws.com:/home/ubuntu/tf-grocery-object-detection/training_demo/exported-models training_demo
```

## Evaluating trained model
We consider three metrics for this usecase: **Precision, Recall and F1-score.**
Precision = TP / (TP+FP)
Recall = TP / (TP+FN)
F1-score = 2PR/(P+R)