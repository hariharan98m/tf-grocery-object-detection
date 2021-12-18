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
python -m pip install --use-feature=2020-resolver .
```
6. Run tests to confirm proper installation.
```bash
python object_detection/builders/model_builder_tf2_test.py
```
### Creating folder structure for custom dataset training.

Use the training_demo