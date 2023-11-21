import cv2, os, numpy as np
import xmltodict, json, random

def xml_to_json(xml_path):
    try:
        with open(xml_path, 'r') as xml_file:
            xml_content = xml_file.read()

        xml_dict = xmltodict.parse(xml_content)

        # Convert OrderedDict to JSON string
        json_data_str = json.dumps(xml_dict, indent=4)

        # Convert JSON string to dictionary
        json_data_dict = json.loads(json_data_str)

    except Exception as e:
        json_data_dict = {}
        print(e)
    return json_data_dict

def get_all_image_paths_with_xml(annotation_dir):
    images_dir = annotation_dir.replace('Annotations', 'Images')
    all_image_paths = []
    for root, dirs, files in os.walk(annotation_dir):
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(root, file)
                image_path = file_path.replace('Annotations', 'Images').replace('.xml', '.jpg')
                all_image_paths.append(image_path)
    return all_image_paths

def get_boxes_from_xml(annotation_path):

    try:
        json_data_dict = xml_to_json(annotation_path)
        objects = json_data_dict['annotation']['object']
        boxes = {}
        for object in objects:
            pts = [[int(pt['x']),int(pt['y'])] for pt in object['polygon']['pt']]
            boxes[object['name']] = get_bbox(pts)
    except Exception as exe:
        boxes = {}
        print(exe)

    return boxes

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    if x3 > x2 or x4 < x1 or y3 > y2 or y4 < y1:
        return 0
    
    elif  (x1<x3 and x2>x4 and y1<y3 and y2>y4):
        return 1
    
    elif  (x3<x1 and x4>x2 and y3<y1 and y4>y2):
        return 1
    

    x5 = max(x1, x3)
    y5 = max(y1, y3)
    x6 = min(x2, x4)
    y6 = min(y2, y4)

    intersection = max(0, x6 - x5 + 1) * max(0, y6 - y5 + 1)
    union = (x2 - x1 + 1) * (y2 - y1 + 1) + (x4 - x3 + 1) * (y4 - y3 + 1) - intersection

    return intersection / union

def is_intersecting(poster_coords, boxes):
    for box in boxes.values():
        if iou(poster_coords, box) > 0:
            return True
    return False


def find_non_overlapping_area(image, bounding_boxes):
    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for box in list(bounding_boxes.values()):
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 255

    non_overlapping_mask = cv2.bitwise_not(mask)
    return non_overlapping_mask

def is_valid(x_start, y_start, poster_w, poster_h, w, h):
    if x_start + poster_w > w or y_start + poster_h > h:
        return False
    else:
        return True

def place_poster(image, poster, non_overlapping_mask, boxes):
    if len(boxes) == 0:
        return image, [0,0,0,0], 'No bounding boxes found'
    
    else:
      result = np.copy(image)
      h, w, _ = image.shape
      poster_h, poster_w, _ = poster.shape

      y_range, x_range = np.where(non_overlapping_mask == 255)
      if len(y_range) != 0 and len(x_range) != 0:
        #print('entered')
        random_pixel = 0
        x_start = x_range[random_pixel]
        y_start = y_range[random_pixel]
        poster_coords = [x_start, y_start, x_start + poster_w, y_start + poster_h]
        counter = 0
        while ((not is_valid(x_start, y_start, poster_w, poster_h, w, h)) or (is_intersecting(poster_coords, boxes))) and random_pixel < len(x_range):
            
                random_pixel += 1
                x_start = x_range[random_pixel]
                y_start = y_range[random_pixel]
                poster_coords = [x_start, y_start, x_start + poster_w, y_start + poster_h]           

                if random_pixel == len(x_range) - 1:
                    poster_h, poster_w = int(poster_h*0.95), int(poster_w*0.95)
                    poster = cv2.resize(poster, (poster_w, poster_h))
                    random_pixel = 0
                    counter += 1
                
                if counter == 15:
                    break
        
        #print('number of resizes: ',counter)

        if counter < 15:
          #print('entered')
          status = 'Poster placed'
          result[y_start:y_start + poster_h, x_start:x_start + poster_w] = poster
        
        else:
          #print('entered')
          status = 'Poster not placed'
          result = result
          x_start, y_start = 0,0
          poster_w, poster_h = 0,0
      
      else:
        status = 'Poster not placed'
        result = result
        x_start, y_start = 0,0
        poster_w, poster_h = 0,0

      return result, [x_start, y_start, x_start+poster_w, y_start+poster_h], status

def get_bbox(pts):
    pts = np.array(pts)
    xmin = np.min(pts[:,0])
    ymin = np.min(pts[:,1])
    xmax = np.max(pts[:,0])
    ymax = np.max(pts[:,1])
    return [xmin, ymin, xmax, ymax]
