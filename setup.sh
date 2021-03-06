# jetson inference
# git clone https://github.com/dusty-nv/jetson-inference.git

# move into pytorch-ssd repo
cd pytorch-ssd

# get pretrained models
# mobilenet v1 SSD
wget -P models https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth
# mobilenetv2-ssd-lite
wget -P models https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth
# vgg-ssd
wget -P models https://storage.googleapis.com/models-hao/vgg16-ssd-mp-0_7726.pth

# get data
wget -P data/helmet https://w251-masknomask.s3.amazonaws.com/helmet.annotations.zip
unzip -q data/helmet/helmet.annotations.zip -d data/helmet
mv data/helmet/annotations data/helmet/Annotations
mv data/helmet/images data/helmet/JPEGImages
# create label file
printf 'helmet,person,head' > data/helmet/labels.txt

# start within the w251-fall2021-project repo
cd ..

