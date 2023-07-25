import numpy as np
import cv2

parent_pairs = [(1, 0), (2, 0), (17, 0), (18, 0), (3, 1), (4, 2), (5, 18), (6, 18), (19, 18), (11, 19), (12, 19), (7, 5), (8, 6), (9, 7), (10, 8), (13, 11), (14, 12), (15, 13), (16, 14), (24, 15), (25, 16), (20, 24), (22, 24), (21, 25), (23, 25)]


keypoint_colors = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
           (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
           (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
           (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
           (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot

score_threshold = 0.1

def get_average_repr(reprs : list):
    average = {}
    #face_size
    average['face_size'] = sum([x['face_size'] for x in reprs])/len(reprs)

    #nose_pos
    nose_pos = [0,0]
    for x in reprs:
        nose_pos[0] += x['nose_pos'][0]
        nose_pos[1] += x['nose_pos'][1]
    nose_pos[0] /= len(reprs)
    nose_pos[1] /= len(reprs)
    average['nose_pos'] = nose_pos

    #relative positions
    
    poses = np.array([x['pose'] for x in reprs])
    average['pose'] = np.average(poses, axis = 0)

    return average

def render_keypoints(cvimage, keypoints, withids):
    for i, point in enumerate(keypoints[:26]):
        position =  (int(point[0] * cvimage.shape[1]), int(point[1] * cvimage.shape[0]))
        cv2.circle(cvimage, position, 4, keypoint_colors[i], -1)
        if withids:
            cv2.putText(cvimage, str(i), position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,  keypoint_colors[i], 2, cv2.LINE_AA)
    return cvimage

def render_representation(cvimage, repr, withids, color = None, lines = 0):
    positions = [[0,0] for x in range(26)]
    positions[0] = repr['pose'][0]
    for pair in parent_pairs:
        i = pair[0]
        point = repr['pose'][i]
        x = point[0] + positions[pair[1]][0]
        y = point[1] + positions[pair[1]][1]
        positions[i] = [x,y]
    #for i in range(26):
    #    positions[i][0] += 0.5
    #    positions[i][1] += 0.5
    size = (cvimage.shape[1], cvimage.shape[0])

    
    left =  int(min(positions, key=lambda x:x[0])[0] * size[0])
    right = int(max(positions, key=lambda x:x[0])[0] * size[0])
    top =   int(min(positions, key=lambda x:x[1])[1] * size[1])
    bottom =int(max(positions, key=lambda x:x[1])[1] * size[1])
    
    cv2.rectangle(cvimage, (left, top), (right, bottom), color, 2)
    for i in range(26):
        positions[i] = (int(positions[i][0] * size[0]), int(positions[i][1]*size[1]))
    for i in range(26):
        if color == None:  color = keypoint_colors[i]
        cv2.circle(cvimage, positions[i], 3,  color, -1)
        if withids:
            cv2.putText(cvimage, str(i), positions[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if lines > 0:
        for pair in parent_pairs:
            if color == None:  color = keypoint_colors[i]
            i = pair[1]
            cv2.line(cvimage, positions[pair[0]], positions[pair[1]], color, lines)
    return cvimage

def render_keypoints_in_frame(cvimage, keypoints, withids, border_size:tuple):
    left, top, right, bottom = border_size
    width = cvimage.shape[1] - right - left
    height = cvimage.shape[0] - bottom - top
    for i, point in enumerate(keypoints[:26]):
        position =  (int(left + point[0] * width), int(top + point[1] * height))
        cv2.circle(cvimage, position, 4, keypoint_colors[i], -1)
        if withids:
            cv2.putText(cvimage, str(i), position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,  keypoint_colors[i], 1, cv2.LINE_AA)
    return cvimage


def get_flattened_pose_repr(json_data):
    flattened_reprs = []
    for data in json_data:
        size = data['size']
        pose = data['keypoints'][0]
        flattened_reprs.append(keypoints2representation(pose, size)['pose'].flatten())
    return np.array(flattened_reprs)


feet = [25, 23, 21, 24, 20, 22]
face = [0, 1, 2, 3, 4, 17, 18]
def distance_list(A, B):
    a = A.reshape((A.size//3, 3))
    b = B.reshape((B.size//3, 3))

    distances = np.linalg.norm(a - b, axis = 1)
    for i in feet:
        distances[i] *= 0.1
    for i in face:
        distances[i] *= 1 / len(face)
    return distances
    #return np.sum(distances)

def distance(A, B):
    return np.sum(distance_list(A, B))

def get_data(image_name, json_data):
    return next(x for x in json_data if x['name'] == image_name)

'''
Distance of person
- Face size
Position of pose in photo(Normalized by photo size)
- Keypoint 0(nose)'s position
The pose itself
- Each keypoint's angle relative to parent keypoint
- Each keypoint's position relative to parent keypoint (Normalized by pose square box)
'''
def keypoints2representation(keypoints, size):
    #face_size
    face_points = keypoints[26:94] + [keypoints[17]]
    left    = min(face_points, key=lambda x:x[0])[0]
    right   = max(face_points, key=lambda x:x[0])[0]
    top     = min(face_points, key=lambda x:x[1])[1]
    bottom  = max(face_points, key=lambda x:x[1])[1]
    face_size = (right-left) * (bottom - top)

    #nose_pos
    nose_pos = keypoints[0][:2]

    #relative positions
    pose = np.array([[key[0] * size[0], key[1] * size[1]] for key in keypoints[:26]])
    #pose = (pose - pose.min(axis = 0)) / (pose.max(axis = 0) - pose.min(axis = 0)) #pose bounding box

    #pose square box
    width= pose.max(axis = 0)[0] - pose.min(axis = 0)[0]
    height= pose.max(axis = 0)[1] - pose.min(axis = 0)[1]
    if width < height:
        pose = (pose - pose.min(axis = 0)) / height
        pose[:,0] += 0.5 - width/height/2
    else:
        pose = (pose - pose.min(axis = 0)) / width
        pose[:,1] += 0.5 - height/width/2

    #make relative    
    for pair in parent_pairs.__reversed__():
        pose[pair[0]] -= pose[pair[1]]
    
    #add score info
    pose = [[point[0], point[1], keypoints[i][2]] for i, point in enumerate(pose.tolist())]
    for i, point in enumerate(pose):
        if point[2] > score_threshold:
            pose[i][2] = 0
        else:
            pose[i] = [0, 0, 1]

    return {'face_size':face_size, 'nose_pos': nose_pos, 'pose':np.array(pose)}
