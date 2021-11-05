# jetson inference
git clone https://github.com/dusty-nv/jetson-inference.git

# opencv
pip install opencv-python

# get pytorch-ssd repo
git clone https://github.com/qfgaohao/pytorch-ssd.git

# move into pytorch-ssd repo
cd pytorch-ssd

# get pretrained models
# mobilenet v1 SSD
wget -P models https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth
# mobilenetv2-ssd-lite
wget -P models https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth
# vgg-ssd
wget -P models https://storage.googleapis.com/models-hao/vgg16-ssd-mp-0_7726.pth

# for cv2 error --> ImportError: libGL.so.1: cannot open shared object file: No such file or directory
apt-get update
apt-get install ffmpeg libsm6 libxext6  -y
