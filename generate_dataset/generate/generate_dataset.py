"""
    Generate Dataset

    1. Converting video to frames
    2. Extracting features
    3. Getting change points
    4. User Summary ( for evaluation )

"""
import os, sys
sys.path.append('../')
#from CNN import ResNet

from cpd_auto import cpd_auto
from tqdm import tqdm
import math
import cv2
import numpy as np
import h5py
import ssl
from CNN import ResNet
ssl._create_default_https_context = ssl._create_unverified_context

class Generate_Dataset:
    def __init__(self, video_path, save_path):
        self.resnet = ResNet()
        #self.gnet = GoogleNet()
        self.dataset = {}
        self.video_list = []
#        self.video_list1 = "/home/swati/Documents/pytorch-vsumm- generate dataset-master/frames/"
        self.video_path = ''
        self.frame_root_path = 'frames'
        self.h5_file = h5py.File(save_path, 'a')
        self._set_video_list(video_path)

    def _set_video_list(self, video_path):
        if os.path.isdir(video_path):

            self.video_path = video_path
            print("Vidpath", video_path)
             
#            frame_root_path = "/home/swati/Documents/pytorch-vsumm- generate dataset-master/videos/frames"
#            self.frame_root_path = frame_root_path
            self.video_list = os.listdir(video_path)
#            print("vp",video_path)
            self.video_list.sort()

        else:
            self.video_path = ''
            self.video_list.append(video_path)

        for idx, file_name in enumerate(self.video_list):
            self.dataset['Video_{}'.format(idx+1)] = {}
            print(self.h5_file.create_group('Video_{}'.format(idx+1)))

    def _extract_feature(self, frame):
#        for frame in enumerate(self.video_list): 
#            print("FRAMES", frame)
        frame = cv2.cvtColor((frame), cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
#        print("frame is ", frame)
        res_pool5 = self.gnet(frame)
        frame_feat = res_pool5.cpu().data.numpy().flatten()
        return frame_feat

    def _get_change_points(self, video_feat, n_frame, fps):
        print("N_frames", n_frame)
        n = n_frame / fps
        m = int(math.ceil(n/2.0))
        K = np.dot(video_feat, video_feat.T)
        change_points, _ = cpd_auto(K, m, 1)
        change_points = 5*(np.concatenate(([0], change_points, [n_frame-1])))
        print("change_points", change_points)
        temp_change_points = []
        for idx in range(len(change_points)-1):
            segment = [change_points[idx]+1, change_points[idx+1]]
#            print("IDX1",change_points[idx])
#            print("IDX2",change_points[idx+1] )
#            print("SEGMENT", segment)
            if idx == len(change_points)-2:
                segment = [change_points[idx], change_points[idx+1]]
                #print("SEGMENT1", segment)


            temp_change_points.append(segment)
        change_points = (np.array((temp_change_points)))

        temp_n_frame_per_seg = []
        for change_points_idx in range(len(change_points)):
            n_frame = (change_points[change_points_idx][1] - change_points[change_points_idx][0])
            temp_n_frame_per_seg.append(n_frame)
            n_frame_per_seg = np.array(list(temp_n_frame_per_seg))

        return change_points, n_frame_per_seg

#    # TODO : save dataset
    def _save_dataset(self):
        pass

    def generate_dataset(self):
        for video_idx, video_filename in enumerate(tqdm(self.video_list)):
            video_path = video_filename
            video_path = self.video_path
            
            if os.path.isdir(video_path):
                video_path = os.path.join(video_path, video_filename)
                print("path is", video_path)

            video_basename = os.path.basename(video_path).split('.')[0]
            print("video_basename is", video_basename)

#            if not os.path.exists(os.path.join(self.frame_root_path, video_basename)):
#                os.mkdir(os.path.join(self.frame_root_path, video_basename))
#            print("PATH", video_path)
            video_capture = cv2.VideoCapture(video_path)
            count = 1
            fps = 1
#            fps = video_capture.get(cv2.CAP_PROP_FPS)
            #print('fps is', fps)
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
#            user_summary = []
#            user_summary.append(9, n_frames)
#            array = np.ones([9, n_frames], dtype = object)
#           
#            print('nframes',  )

            frame_list = []
            picks = [] 
            video_feat = None
            video_feat_for_train = None
            success = False
            for frame_idx in tqdm(range(n_frames-1)):
                success, frame = video_capture.read()
                #print("Success", success)
                if success:
#                    print("F",frame)
                    frame_feat = self._extract_feature(frame)
                    

                if frame_idx % 10 == 0:
                        picks.append(frame_idx)
                        print("length frame_idx", len(picks))
#                        
                        if video_feat_for_train is None:
                            video_feat_for_train = frame_feat
                        else:
                            video_feat_for_train = np.vstack((video_feat_for_train, frame_feat))
        
                        if video_feat is None:
                            video_feat = frame_feat
                        else:
                            video_feat = np.vstack((video_feat, frame_feat))

                img_filename = "/frame%d.jpg" % count
                #print("PATH",("/home/swati/Documents/pytorch-vsumm- generate dataset-master/frames1/"+ video_basename + img_filename))
                cv2.imwrite(os.path.join("/home/swati/Desktop/Master_Project/generate_dataset/frames1/" + video_basename + img_filename), frame)
                #path_check = os.path.join("/home/swati/Documents/pytorch-vsumm- generate dataset-master/frames1/" + video_basename + img_filename)
                count+= 1
#                else:
#                    print("frame is..")
#                    break

#            video_capture.release()

            

            
            if success:
                change_points, n_frame_per_seg = self._get_change_points(video_feat,len(picks), fps)
#                array = np.ones(9, n_frames, dtype = object)
#                
                print('video_feat_for_train',type(video_feat_for_train))
                print('video_feat_for_train_shape',(video_feat_for_train.shape))
#                video_feat_for_train = (video_feat_for_train.reshape(-1,1024))
                print('video_feat_for_train_reshape', video_feat_for_train.shape)
                self.h5_file['Video_{}'.format(video_idx+1)]['features'] = video_feat_for_train
                print('video_feat_for_train',video_feat_for_train)
#                video_feat_for_train = (video_feat_for_train.maxshape(None,1048))
                print('video_feat_for_train_reshape', video_feat_for_train.shape)
                self.h5_file['Video_{}'.format(video_idx+1)]['picks'] = np.array(list(picks))
                self.h5_file['Video_{}'.format(video_idx+1)]['n_frames'] = n_frames
                self.h5_file['Video_{}'.format(video_idx+1)]['fps'] = fps
                self.h5_file['Video_{}'.format(video_idx+1)]['change_points'] = change_points
                self.h5_file['Video_{}'.format(video_idx+1)]['n_frame_per_seg'] = n_frame_per_seg
#                self.h5_file['Video_{}'.format(video_idx+1)]['user_summary'] = user_summary

            print("Directory is ", os.getcwd())


if __name__ == "__main__":
    gen = Generate_Dataset('Master_Project/generate_dataset/VideosAction/', 'final.h5')
    gen.generate_dataset()
    gen.h5_file.close()