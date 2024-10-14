<<<<<<< HEAD
# MSAED
Multi-scale Attention for document image enhancement
=======
Requirement package:
pytorch==1.6.0
torchvision==0.7.0
visdom==0.1.8.9

# Prepare data:
See file `make_data_v4.py` to create training data. 
Sample data can found in Sample-data. Actually, training image has size 256*512 

# training
 `python train.py --dataroot ../data_256_512 --netG resnet_6blocks --model attngatedbamweight  --name resnet6blocks_attngated_bam --gpu_ids 3 --batch_size 4 --input_nc 1 --output_nc 1`

The dataroot contains only few sample. You need to make full samples (~2500)

# testing
`python test.py --dataroot ../test --netG resnet_6blocks --model attngatedbamweight  --name resnet6blocks_attngated_bam --gpu_ids -1 --epoch 5`

The dataroot contains some sample. Add more image you want to test
This zip file included bestcheckpoint at epoch 375
>>>>>>> 5ab6820 (Add local README.md)
