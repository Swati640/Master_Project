python create_split.py  --save-dir datasets  --save-name splits --num-splits 5 --train-percent 0.8
python main.py --dataset datasets/UCF101.h5 -s datasets/UCF101.json --gpu 0
python summary2video.py --path summe-split1/result.h5  --frm-dir ApplyEyeMakeup -i 1 --fps 7 --width 640 --height 480
python train.py --dataset_path data/UCF-101-frame
