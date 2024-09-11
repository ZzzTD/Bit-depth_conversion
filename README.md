# Bit-depth_conversion
This is the code implementation of "Unsupervised conversion method of high bit depth remote sensing images using contrastive learning"

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Update log

9/11/2024: Added related codes.

### Train and Test

- Train the model:
```bash
python train.py --dataroot XXX --name XXX
```

- Test the model:
```bash
python test.py --dataroot XXX --name XXX
```

### Datesets
All the data mentioned in the article has been uploaded to Baidu Cloud, link is:https://pan.baidu.com/s/1NVu1yWH7cnFf56iHuakJZw(cmbn) 


### Acknowledgments
Our code is developed based on [contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) 
