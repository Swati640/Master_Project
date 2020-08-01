import h5py
import sys
import cv2
import os
import os.path as osp
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True, help="path to h5 result file")
parser.add_argument('-d', '--frm-dir', type=str, required=True, help="path to frame directory")
parser.add_argument('-i', '--idx', type=int, default=1, help="which key to choose")
parser.add_argument('--fps', type=int, default=7, help="frames per second")
parser.add_argument('--width', type=int, default=640, help="frame width")
parser.add_argument('--height', type=int, default=480, help="frame height")
# parser.add_argument('--save-dir', type=str, default='datasets', help="directory to save")
# parser.add_argument('--save-name', type=str, default='', help="video name to save (ends with .mp4)")
args = parser.parse_args()
def frm2video(frm_dir, summary, vid_writer):
    
    for idx, val in enumerate(summary):
        if val == 1:
            frm_name = str(idx) + '.jpg'
#            print("FRAME_NAME", frm_name)
            frm_path = osp.join(frm_dir) + "/"+ frm_name
            print("PATH", frm_path)
#            print("DIR", frm_dir)
#            print("PATH", frm_path)
            frm = cv2.imread(frm_path)
#            print("frame", frm)
            frm = cv2.resize(frm, (args.width, args.height))
            vid_writer.write(frm)

if __name__ == '__main__':
    # if not osp.exists(args.save_dir):
    #     print(os.mkdir(args.save_dir))
#    vid_writer = cv2.VideoWriter(
#    osp.join(args.save_dir, args.save_name),
#    cv2.VideoWriter_fourcc(*'MP4V'),
#    args.fps,
#    (640,480),
#    )
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_writer= cv2.VideoWriter('video.avi',fourcc, 20.0, (640,480))
##    vid_writer = cv2.VideoWriter(
#        osp.join(args.save_dir, args.save_name),
#        cv2.VideoWriter_fourcc(*'X264'),
#        args.fps,
#        (640,480),
#    )
#    h5_file_name = (sys.argv[1])
#    f= h5py.File(h5_file_name, "r")
#    for key in f.keys():
#        print("{} : {}".format(key, f[key]['video_name'].value))

    h5_res = h5py.File(args.path, 'r')
    key = list(h5_res.keys())[args.idx]
#    print("KEY", key)
    summary = h5_res[key]['machine_summary'][...]
    h5_res.close()
    frm2video(args.frm_dir, summary, vid_writer)
    vid_writer.release()