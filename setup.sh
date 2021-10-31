# run on EC2 instance with NVIDIA Deep Learning AMI

# start up a Docker container
# docker run --gpus all -it --rm -p 8888:8888 --shm-size=2048m -v "$PWD":/workspace/ nvcr.io/nvidia/pytorch:21.09-py3

# jetson inference
git clone https://github.com/dusty-nv/jetson-inference.git

# opencv
pip install opencv-python

# get pytorch-ssd repo
git clone https://github.com/qfgaohao/pytorch-ssd.git