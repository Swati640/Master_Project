# Video Summarization and Action Recognition

## Docker

Pull the image from docker hub for generating the dataset on your own dataset(It Preprocess the videos' feature to H5 file). This image has all the required packages. It has the dependency on python 2.7 for certain libraries.
```bash 
docker pull swati640/generate_dataset:latest
```
Pull this image from docker hub for the video summary and action recognition. It has all the required packages.
```bash 
docker pull swati640/summary:latest
```
## Download the data and preprocess it
Download the data from [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php). The complete work is done on the subset of UCF101. Randomly 21 videos are selected from the complete dataset. 

### Create the HDF5 file from videos

``` bash
python generate_dataset.py
```
## Make splits
```bash
python create_split.py  --save-dir datasets  --save-name splits --num-splits 5 --train-percent 0.8
```
As a result, the dataset is randomly split for 5 times, which are saved as json file. Train and test codes are written in `main.py`. To see the detailed arguments, please do `python main.py -h`.

## How to train
```bash
python main.py --dataset datasets/UCF101.h5 -s datasets/UCF101.json --gpu 0
```
## Visualize summary
You can use `summary2video.py` to transform the binary `machine_summary` to real summary video. You need to have a directory containing video frames. The code will automatically write summary frames to a video where the frame rate can be controlled.
```bash
python summary2video.py --path summe-split1/result.h5  --frm-dir ApplyEyeMakeup -i 1 --fps 7 --width 640 --height 480
```
