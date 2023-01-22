from utils.utilities import get_cfg,count_parameters
from model.network import parsingNet
from evaluation.eval_wrapper import eval_lane
import torch
import os
from demo.demo_utils.utilities import build_meta_file
import time
import cv2
import matplotlib.pyplot as plt
import warnings

def display(lane_file, image_file):
    lanes = open(lane_file).readlines()
    im = cv2.imread(image_file)
    print(image_file)
    colors = [(255,0,0), (0,255,0), (0,0,255)]
   
    for i,lane in enumerate(lanes):
        lane = lane.split(" ")
        lane = [(int(x),int(y)) for x,y in zip(lane[0::2], lane[1::2])]
        [cv2.circle(im, center=(x,y), radius=1, color=colors[i], thickness=2) for x,y in lane]

    cv2.imshow("output", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ssh_send(file_to_send):
    print("sending : ", file_to_send)
   

def send(work_dir, meta_list_file, done):
    saved_lanes_folder = os.path.join(work_dir, "culane_eval_tmp")
    meta_list = open(meta_list_file, "r").readlines()
    for meta_file in meta_list:
        done.append(os.path.split(meta_file)[1])
        related_lane_file = meta_file.split(".")[0] + ".lines.txt"                    
        if os.path.exists(os.path.join(saved_lanes_folder, related_lane_file)):
            ssh_send(os.path.join(saved_lanes_folder, related_lane_file))
            done.append(os.path.split(related_lane_file)[1])
            
        

def check_points(path, meta_list_file, work_dir, seq_len):        
    saved_lanes_folder = os.path.join(work_dir, "culane_eval_tmp")
    meta_list = open(meta_list_file, "r").readlines()
    for meta_file in meta_list:
        related_lane_file = meta_file.split(".")[0] + ".lines.txt"                    
        if os.path.exists(os.path.join(saved_lanes_folder, related_lane_file)):
            display(os.path.join(saved_lanes_folder, related_lane_file), os.path.join(path, meta_file.rstrip()))

def flush_queue(root_dataset, work_dir, done):
    saved_lanes_folder = os.path.join(work_dir, "culane_eval_tmp", "image_frames")
    images_foldes = os.path.join(root_dataset, "image_frames")

    list_dir_saved_lanes = os.listdir(saved_lanes_folder)
    list_dir_images_folder = os.listdir(images_foldes)

    for saved_lanes in list_dir_saved_lanes:        
        if saved_lanes in done:
            os.remove(os.path.join(saved_lanes_folder, saved_lanes))
           
    for image in list_dir_images_folder:
        if image.split(".")[0] + ".lines.txt" in done:
            os.remove(os.path.join(images_foldes, image))


def continous_detection(done):
    while(1):
        print("waiting for images ... ")
        build_meta_file(cfg)
       
        start_time = time.time()    
    
        eval_lane(net, cfg.train.path, cfg.test.test_work_dir, cfg.model.griding_num, cfg.model.seq_len, mode='test', device=cfg.model.device) 
    
        end_time = time.time()

        print("running evaluation time : ", str(end_time - start_time) + " seconds")
   
        send(work_dir=cfg.test.test_work_dir, meta_list_file=cfg.test.test_list_path, done=done)

        print(done)
        flush_queue(root_dataset=cfg.train.path, work_dir=cfg.test.test_work_dir, done=done)



if __name__ == "__main__":
   
    cfg = get_cfg()

    net=parsingNet(cfg).cuda(cfg.model.device)
    
    state_dict = torch.load(cfg.save.weights_path, map_location='cpu')
    state_dict_model=state_dict['model']

    net.load_state_dict(state_dict_model)

    done = []
    continous_detection(done)
    

    

   


