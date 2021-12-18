# Grocery objects detection and classification

### Setting up the Environment

1. Install tensorflow
```bash
pip install tensorflow
```
2. Clone the official TF detection models.
```bash
git clone https://github.com/tensorflow/models.git
```
3. Install protobuf for tf.
```bash
cd models/research
!protoc object_detection/protos/*.proto --python_out=.
```