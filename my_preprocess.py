import numpy as np

def bbox_from_openpose(keypoints, rescale=1.2, detection_thresh=0.01):
    """Get center and scale for bounding box from openpose detections."""
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)
    # adjust bounding box tightness
    bbox_size = bbox_size * rescale
    bbox = [
        center[0] - bbox_size[0]/2, 
        center[1] - bbox_size[1]/2,
        center[0] + bbox_size[0]/2, 
        center[1] + bbox_size[1]/2,
        keypoints[valid, 2].mean()
    ]
    return bbox

def convert_to_format(input_string):
    # Split the string into individual float values
    values = list(map(float, input_string.split()))
    
    # Group every three values into sublists
    formatted_list = [values[i:i+3] for i in range(0, len(values), 3)]
    
    return formatted_list

def my_load_data(pose):
    pid = 0
    out = []
    #print("aaa")
    pose = pose.replace('[', '')
    pose = pose.replace(']', '')
    pose = pose.replace('\n', '')
    #print(pose)
    keypoints = convert_to_format(pose)
    #print("kk",keypoints)
    for i in range(len(keypoints)//24):
        annot = {
                'bbox': bbox_from_openpose(np.array(keypoints)),
                'personID': pid + i,
                'keypoints': keypoints,
                'isKeyframe': False
            }
    out.append(annot)
    #print("aaa")
    return out

def create_annot_file(img):
    height, width = img.shape[0], img.shape[1]
    annot = {
        'filename':'image',
        'height':height,
        'width':width,
        'annots': [],
        'isKeyframe': False
    }
    return annot

def convert_from_openpose(img, pose):
    # convert the 2d pose from openpose
    annot = create_annot_file(img)
    annot['annots'] = my_load_data(pose)
    return annot