# w251-fall2021-project
Final Project for Fall 2021

1. Start up EC2 instance (g4pn.xlarge) with NVIDIA Deep Learning AMI and 500 GB - EBS storage. If you are connecting to a stopped instance, skip to step 4.

2. Clone project repository  
```git clone https://github.com/janehung04/w251-fall2021-project.git```

3. `cd w251-fall2021-project/`

3. Run [./setup.sh](setup.sh) which contains instructions for repo download.

4. Open Docker container  
```docker run --gpus all -it --rm -p 8888:8888 --shm-size=2048m -v "$PWD":/data/ nvcr.io/nvidia/pytorch:21.09-py3```

5. Move to the right directory in the Docker container
```cd ../data```

6. opencv + pandas etc.
`pip install -r requirements.txt`

7. convert PNG to JPG
`cd pytorch-ssd`
`python png_to_jpg.py data/helmet/`

8. get the right imageset
`python vision/datasets/generate_vocdata.py labels.txt /data/pytorch-ssd/data/helmet`

9. for cv2 error --> ImportError: libGL.so.1: cannot open shared object file: No such file or directory
`apt-get update`
`apt-get install ffmpeg libsm6 libxext6  -y`

10. Retrain using these images  
```python train_ssd.py --dataset_type voc --datasets /data/pytorch-ssd/data/helmet --validation_dataset /data/pytorch-ssd/data/helmet --net mb1-ssd --pretrained_ssd models/mobilenet-v1-ssd-mp-0_675.pth --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 100 --base_net_lr 0.001  --batch_size 5```

