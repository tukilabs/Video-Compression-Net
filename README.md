# Video-Compression-Net
This project presents the Neural architecture to compress videos (sequence of image frames) along with the pre-trained models. Our work is inspired by [DVC](https://github.com/GuoLusjtu/DVC) and we use [tensorflow-compression](https://github.com/tensorflow/compression/) for bitrate estimation and entropy compression. Compression is realized in terms of actual file size.

## Installation
For installation, simply run the following command:
```bash
pip install -r requirements.txt
```
for GPU support, replace the tensorflow==1.15.0 line in requirements.txt with tensorflow-gpu==1.15.0 .

**Note: Precompiled packages for tensorflow-compression are currently only provided for Linux (Python 2.7, 3.3-3.6) and Darwin/Mac OS (Python 2.7, 3.7). For windows please refer [this](https://github.com/tensorflow/compression/blob/master/README.md).**

## Pre-trained Models:
Pre-trained models are available at [checkpoints](https://github.com/tukilabs/Video-Compression-Net/tree/master/checkpoints). The models suffixed with "msssim" are the ones that are optimized with MS-SSIM while the rest are optimized with PSNR. The Integer in the filename denotes the lambda (weight assigned to distortion compared to the bitrate). Higher the value of lambda lower will be the distortion and higher will be the bitrate.

## Compression
Run the following command and follow the instructions:
```bash
python compress.py -h 
```
For example,
```bash
python compress.py -i demo/input/ -o demo/compressed/ -m checkpoints/videocompressor1024.pkl -f 101
```
The execution compresses the frames in `demo/input/` to compressed files in `demo/compressed/`. 

**Note: Right now, our work is only applicable to the RGB frames with height and width <= 512 and both being multiple of 16. To compress the frames with higher resolution, increase the depth of the pyramid (change `self.netBasic = [Basic() for _ in range(5)]` in [network.py](https://github.com/tukilabs/Video-Compression-Net/blob/master/utils/network.py) to `self.netBasic = [Basic() for _ in range(6)]` and train the network. Needless to say, higher resolution images require more time to train, compress and decompress**

## Reconstruction
Run the following command and follow the instructions:
```bash
python decompress.py -h 
```
For example,
```bash
python decompress.py -i demo/compressed/ -o demo/reconstructed -m checkpoints/videocompressor1024.pkl -f 101
```
The execution will reconstruct the original frames in `demo/reconstructed/` with some compression artifacts.

## Training your own model
We trained the network with [vimeo-septuplet](http://toflow.csail.mit.edu/index.html#septuplet) dataset.To download the dataset, run the script `download_dataset.sh` as:
```bash
sh download_dataset.sh
```
[Here](https://github.com/tukilabs/Video-Compression-Net/tree/master/vimeo_septuplet/sequences), we provide the small portion of the large dataset, to present the dataset outlook.
You can train your own model by simply executing the following command and following the instructions:
```bash
python train.py -h 
```
For training the dataset structure should be same as [vimeo-septuplet](https://github.com/tukilabs/Video-Compression-Net/tree/master/vimeo_septuplet/sequences) structure, otherwise you should write your own data-parser to the network.

## Evaluation
We perform compression and reconstruction in a single file `test.py` for evaluation. To evaluate the compression and distortion, execute:
```bash
python test.py -h
```
and follow the instructions.
For example,
```bash
python test.py -m checkpoints/videocompressor256-msssim.pkl
```

## Demo
![](https://github.com/tukilabs/Video-Compression-Net/blob/master/demo/demo.gif)<br/>
***Note: The compression and reconstruction without GPU will be slower than the above demonstration.***

## Authors
[Prasanga Dhungel](https://github.com/PrasangaDhungel)<br/>
[Prashant Tandan](https://github.com/Prashant528)<br/>
[Sandesh Bhusal](https://github.com/sandeshbhusal)<br/>
[Sobit Neupane](https://github.com/sobitneupane)<br/>
