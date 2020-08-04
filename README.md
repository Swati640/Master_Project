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
