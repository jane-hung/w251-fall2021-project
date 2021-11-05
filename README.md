# w251-fall2021-project
Final Project for Fall 2021

1. Start up EC2 instance (g4pn.xlarge) with NVIDIA Deep Learning AMI and 500 GB - EBS storage.

2. Ensure images are stored in the host machine
```wget https://w251-masknomask.s3.amazonaws.com/helmet.zip```
```mkdir data/helmet```
```unzip -q helmet.zip -d data/helmet```

2. Open Docker container  
```docker run --gpus all -it --rm -p 8888:8888 --shm-size=2048m -v "$PWD":/data/ nvcr.io/nvidia/pytorch:21.09-py3```

3. Run [setup.sh](setup.sh) which contains instructions for repo download.

4. Retrain using these images  
```python train_ssd.py --dataset_type open_images --datasets "$PWD"/data/open_images --net mb1-ssd --pretrained_ssd models/mobilenet-v1-ssd-mp-0_675.pth --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 100 --base_net_lr 0.001  --batch_size 5```

