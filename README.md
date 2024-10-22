# MSAED
Multi-scale Attention for Document Image Enhancement [MSAED](https://link.springer.com/chapter/10.1007/978-3-030-86549-8_26) 
The official code for implemention of paper at ICDAR 2021.
## Requirements
- `pytorch==1.6.0`
- `torchvision==0.7.0`
- `visdom==0.1.8.9`

## Prepare Data
See the file `make_data_v4.py` to create training data. Sample data can be found in `Sample-data`. The training images have a size of 256x512.

## Training
To start training, run the following command:

```bash
python train.py --dataroot ../data_256_512 --netG resnet_6blocks --model attngatedbamweight --name resnet6blocks_attngated_bam --gpu_ids 3 --batch_size 4 --input_nc 1 --output_nc 1
```
## Testing
To test the model, run:
```bash
python test.py --dataroot ../test --netG resnet_6blocks --model attngatedbamweight --name resnet6blocks_attngated_bam --gpu_ids -1 --epoch 5
```
The dataroot contains some samples. Add more images you want to test. This zip file includes the best checkpoint at epoch 375.
### Infos
This code is heavily based from [Pix2pix](https://github.com/phillipi/pix2pix). If you encounter any issues or have further questions, feel free to ask!
