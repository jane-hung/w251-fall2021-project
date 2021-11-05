# w251-fall2021-project
Final Project for Fall 2021

1. Start up EC2 instance (g4pn.2xlarge) with NVIDIA Deep Learning AMI and 500 GB - EBS storage.

2. Open Docker container  
```docker run --gpus all -it --rm -p 8888:8888 --shm-size=2048m -v "$PWD":/data/ nvcr.io/nvidia/pytorch:21.09-py3```

3. Run [setup.sh](setup.sh) which contains instructions for repo download. Can create new docker image with these git repos cloned?

4. Get data

```mkdir /data/helmets```

```cd /data/helmets```

```wget https://w251-masknomask.s3.amazonaws.com/helmet.zip```

```unzip helmet.zip```

5. Retrain using these images
```python train_ssd.py --dataset_type open_images --datasets /data/helmets --net mb1-ssd --pretrained_ssd models/mobilenet-v1-ssd-mp-0_675.pth --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 100 --base_net_lr 0.001  --batch_size 5```

