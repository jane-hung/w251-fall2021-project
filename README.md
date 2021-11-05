test
# w251-fall2021-project
Final Project for Fall 2021

1. Open Docker container  
```docker run --gpus all -it --rm -p 8888:8888 --shm-size=2048m -v "$PWD":/workspace/ nvcr.io/nvidia/pytorch:21.09-py3```

2. Run [setup.sh](setup.sh) which contains instructions for repo download. Can create new docker image with these git repos cloned?

3. Try tutorial on guns  
```cd pytorch-ssd```  
```wget -P models https://storage.googleapis.com/models-hao/gun_model_2.21.pth```  
```wget -P models https://storage.googleapis.com/models-hao/open-images-model-labels.txt```  
```python run_ssd_example.py mb1-ssd models/gun_model_2.21.pth models/open-images-model-labels.txt ~/Downloads/big.JPG```
? What is the big.JPG file?

4. Download more images  
```python open_images_downloader.py --root "$PWD"/data/open_images --class_names "Handgun,Shotgun" --num_workers 20```

5. Retrain using these images
```python train_ssd.py --dataset_type open_images --datasets ~/data/open_images --net mb1-ssd --pretrained_ssd models/mobilenet-v1-ssd-mp-0_675.pth --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 100 --base_net_lr 0.001  --batch_size 5```

