from PIL import Image
import time, cv2, math
import numpy as np
import matplotlib.pyplot as plt
from Detector import *
detect=Detector()
import math
import torch
from deepsort.deep_sort import preprocessing, nn_matching
from deepsort.deep_sort.detection import Detection
from deepsort.deep_sort.tracker import Tracker
import copy
import deepsort.tools.generate_detections as gdet

def get_pt_from_box(box):
   return [(box[0]+box[2])/2,(box[1]+box[3])/2]


max_cosine_distance = 0.1
nn_budget = 4
nms_max_overlap = 0.5

# initialize deep sort
model_filename = r"C:\Users\balub\OneDrive\Desktop\CMU\detectron2\deepsort\model_data\mars-small128.pb"
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
# calculate cosine distance metric
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# initialize tracker
tracker = Tracker(metric, max_iou_distance=0.8, max_age=120, n_init=1)
#######################################
bbox_distance_threshold = 10  
counter_threshold = 3
#######################################
def voc2coco(box):## (xmin,ymin,xmax,ymax) -> (xmin,ymin,w,h)
  return [box[0],box[1],box[2]-box[0],box[3]-box[1]]

def compute_iou(boxes1, boxes2):
  box_in_box = False
  x1_min, y1_min, x1_max, y1_max = boxes1
  x2_min, y2_min, x2_max, y2_max = boxes2

  if (x1_min < x2_min and y1_min < y2_min and x1_max > x2_max and y1_max > y2_max) or (x1_min > x2_min and y1_min > y2_min and x1_max < x2_max and y1_max < y2_max):
      box_in_box = True

  boxes1=torch.tensor(boxes1)
  boxes2=torch.tensor(boxes2)  
  # Compute intersection coordinates
  intersection_xmin = torch.max(boxes1[0], boxes2[0])
  intersection_ymin = torch.max(boxes1[1], boxes2[1])
  intersection_xmax = torch.min(boxes1[2], boxes2[2])
  intersection_ymax = torch.min(boxes1[3], boxes2[3])

  # Compute intersection area
  intersection_area = torch.clamp(intersection_xmax - intersection_xmin, min=0) * torch.clamp(intersection_ymax - intersection_ymin, min=0)

  # Compute union area
  boxes1_area = abs((boxes1[2] - boxes1[0]) * (boxes1[3] - boxes1[1]))
  boxes2_area = abs((boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1]))
  union_area = boxes1_area + boxes2_area - intersection_area

  # Compute IoU
  iou = intersection_area / union_area

  return iou.cpu().numpy().tolist(), box_in_box

def multibox_correction(dict1,dict2):
   """
   correction introduced based on IoU
   dict1: initial tracker
   dict2: new tracker after compare
   """
   modified_track = {}
   modified_initial_track = {}
   initial_IDs, new_tracker_IDs = list(dict1.keys()), list(dict2.keys())
   new_tracker_IDs_copy = copy.deepcopy(new_tracker_IDs)
   initial_tracker_IDs_copy = copy.deepcopy(initial_IDs)
   new_IDs_in_new_tracker = list(set(new_tracker_IDs) - set(initial_IDs))
   print('difference: ', new_IDs_in_new_tracker)
   for new_ID in new_IDs_in_new_tracker:
      a = -123
      for initial_ID in initial_IDs:
         iou, box_in_box = compute_iou(dict2[new_ID], dict1[initial_ID])
         # if new_ID == 11 and initial_ID == 7:
            # print('IOU: ',iou, box_in_box)
         # if initial_ID == a:
            # continue
         print('new_ID: ',new_ID,' initial_ID: ', initial_ID)
         print('iou: ',iou)
         if iou > 0.1 or box_in_box:
            print(new_tracker_IDs_copy)
            
            #delete the box
            if initial_ID in initial_tracker_IDs_copy:
               initial_tracker_IDs_copy.remove(initial_ID)
            if initial_ID in new_tracker_IDs_copy:
               new_tracker_IDs_copy.remove(initial_ID)
            # a = initial_ID
         
         
         
      continue

   for ID in initial_tracker_IDs_copy:
      modified_initial_track[ID] = dict1[ID]
   
   for ID in new_tracker_IDs_copy:
      modified_track[ID] = dict2[ID]
   
   print('intial track after iou corr: ',list(modified_initial_track.keys()))
   return modified_track, modified_initial_track


def make_detection(dict1,dict2, difference):
   """
   dict:{Track_ID1 : bbox1, Track_ID2 : bbox2,.........}
   dict1->track from previous frame
   dict2->track from current frame
   """
   keys1,keys2 = list(dict1.keys()), list(dict2.keys())
   # print('len keys1 ',len(keys1))
   # print('len keys2 ',len(keys2))
   if len(keys1)!= 0 and len(keys2)!= 0 and difference!={}:
      missing_IDs = [key for key in keys1 if key not in keys2]
      
      tot_displ = [0,0]
      num = 0
      # print('dict1: ',dict1,' ','dict2: ',dict2)
      for key in keys1:
         if key not in missing_IDs:
            num+=1
            pt1, pt2 = get_pt_from_box(dict1[key]), get_pt_from_box(dict2[key])
            # print('pt1: ',pt1, 'pt2: ',pt2)
            # print('displ: ',tot_displ)
            tot_displ[0] += pt2[0] - pt1[0]
            tot_displ[1] += pt2[1] - pt1[1]
            # print('pt1: ',pt1, 'pt2: ',pt2)
            # print('displ: ',tot_displ)
      
      avg_displ = [tot_displ[0]/num,tot_displ[1]/num]

      if abs(avg_displ[0])<30 and abs(avg_displ[1])<30:
         for ID in missing_IDs:
            missing_det_pos = [0,0]
            missing_det_bboxes_prev = dict1[ID]
            mis_pt1_x, mis_pt1_y = get_pt_from_box(missing_det_bboxes_prev)  
            missing_det_pos[0], missing_det_pos[1] = mis_pt1_x + avg_displ[0], mis_pt1_y + avg_displ[1]
            missing_det_prev_dims = [(missing_det_bboxes_prev[2]-missing_det_bboxes_prev[0])//2,(missing_det_bboxes_prev[3]-missing_det_bboxes_prev[1])//2]
            missing_det_bboxes_new = [missing_det_pos[0]-missing_det_prev_dims[0],missing_det_pos[1]-missing_det_prev_dims[1],
                                    missing_det_pos[0]+missing_det_prev_dims[0],missing_det_pos[1]+missing_det_prev_dims[1]]
            dict2[ID] = missing_det_bboxes_new

   return dict2      

def compare_tracks(dict1,dict2,dict3):
   """
   dict:{Track_ID1 : bbox1, Track_ID2 : bbox2,.........}
   dict1->track from previous frame
   dict2->track from current frame
   dict3->track from previous frame with dimensions of each bbox
   """
   keys1,keys2,keys3 = list(dict1.keys()), list(dict2.keys()), list(dict3.keys())
   if len(keys1)!=0 and len(keys2)!=0:

         new_boxes={}
        
         for key1 in keys1:
            for key2 in keys2:
               if key1==key2:

                  distance=math.dist(get_pt_from_box(dict1[key1]),get_pt_from_box(dict2[key2]))
                  # print('the points are: ',get_pt_from_box(dict1[key1]),' ',get_pt_from_box(dict2[key2]))
                  # print('this is the distance: ',distance)
                  if distance < bbox_distance_threshold:
                     dict2[key2] = dict1[key1]
   
                 
         for key2 in keys2:
            for key3 in keys3:
               if key2 == key3:
                  x_old,y_old = get_pt_from_box(dict2[key2])
                  w,h = dict3[key3]
                  new_boxes[key2] = [x_old-w, y_old-h, x_old+w, y_old+h]
               else:
                  new_boxes[key2] = dict2[key2]
              
   return new_boxes
                  
video_path = r"C:\Users\balub\OneDrive\Desktop\DEEP SORT\PUNCH VIDEO\sample punch.mp4"

# Load the video
video = cv2.VideoCapture(video_path)

# Get video properties
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
i = 0
counter_var = 0
# Process each frame and perform inference
frames_matrix=[]
initial_tracker = {}
while video.isOpened():
    ret, frame = video.read()
    i+=1
   #  if i==110:
   #     break
    print('\nframe: ',i)
    if not ret:
        break

    start_time=time.time()
    # Perform inference on the frame
    predicted = detect.img(frame)
    data = predicted['instances']
    boxes=data.pred_boxes.tensor.tolist()
    classes=data.pred_classes.tolist()
    scores = data.scores.tolist()
    boxes_pred = [box for box in boxes if classes[boxes.index(box)]==1]
    scores_pred = [score for score in scores if classes[scores.index(score)]==1]

    boxes_in_coco = [voc2coco(box) for box in boxes_pred]
    features = encoder(frame, boxes_in_coco)
    names = []
    for itera,box in enumerate(boxes_in_coco): 
      names.append('Screw')
    names = np.array(names)
    print(names)
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes_in_coco, scores, names, features)]
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    tracker.predict()
    tracker.update(detections)
    
    # update tracks
    new_tracker = {}
    tracker_with_dims = {}
    for track in tracker.tracks:
      if not track.is_confirmed() or track.time_since_update > 1:
        continue 
      confirmed = [track for track in tracker.tracks if track.is_confirmed()]
      # print('confirmed: ',len(confirmed))
      bbox = track.to_tlbr()#get_pt_from_box
      new_tracker[track.track_id] = bbox.tolist()
      class_name = track.get_class()
      if len(initial_tracker.values())!=0 and len(new_tracker.values())!=0:
        if track.track_id in list(initial_tracker.keys()):
          prev_bbox = initial_tracker[track.track_id]
          tracker_with_dims[track.track_id] = [(prev_bbox[2] - prev_bbox[0])//2,(prev_bbox[3] - prev_bbox[1])//2]
      else:
         new_tracker_boxes_after_modification=[]

    fps = 1.0 / (time.time() - start_time)

    # draw bbox on screen
    
    if len(initial_tracker.values())!=0 and len(new_tracker.values())!=0:
      new_tracker_after_modification = compare_tracks(initial_tracker,new_tracker,tracker_with_dims)   
      missing_IDs = [ID for ID in list(initial_tracker.keys()) if ID not in list(new_tracker_after_modification.keys())]
      new_tracker_after_modification, initial_tracker = multibox_correction(initial_tracker, new_tracker_after_modification)

      initial_tracker_IDs, new_tracker_after_modification_IDs = list(initial_tracker.keys()), list(new_tracker_after_modification.keys())
      difference = set(initial_tracker_IDs)- set(new_tracker_after_modification_IDs)
      if difference!={} and counter_var<counter_threshold:
         new_tracker_after_modification = make_detection(initial_tracker, new_tracker_after_modification, difference)
         counter_var += 1   
         if counter_var> counter_threshold:
            counter_var = 0
     
      # print('new track aftrer compaere and add', new_tracker_after_modification)
      
      
      new_tracker_boxes_after_modification = list(new_tracker_after_modification.values())
      new_tracker_boxes_after_modification_IDs = list(new_tracker_after_modification.keys())
      new_tracker = new_tracker_after_modification
      for id,bbox in zip(new_tracker_boxes_after_modification_IDs,new_tracker_boxes_after_modification):
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0), 4)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), (0,255,0), -1)
        cv2.putText(frame, class_name + "-" + str(id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
        # print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

    initial_tracker = new_tracker
    result = np.asarray(frame)
    result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # cv2_imshow("Output Video", result)
    output_video.write(result)

## printout list of frames matrix
video.release()
output_video.release()
