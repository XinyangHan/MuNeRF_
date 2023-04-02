from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import json
import numpy as np
import pdb
from numpyencoder import NumpyEncoder

def interpolate_pose_new(pose1, pose2, inter_num=10):
    """ pose1 and pose2 are 3x4 matrix """
    def interpolate(a, b, idx):
        return a + idx*(b-a)/inter_num
    new_cameras = []
    pdb.set_trace()

    rots = np.concatenate((pose1[np.newaxis,:3,:3], pose2[np.newaxis,:3,:3]), axis=0)
    key_rots = R.from_matrix(rots)
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    times = np.arange(0,1,1/inter_num)
    interp_rots = slerp(times)
    interp_rots_mat = interp_rots.as_matrix()
    for idx in range(inter_num):
        inter_pos = interpolate(pose1[:3,3], pose2[:3,3], idx)
        inter_rot = interp_rots_mat[idx]
        
        new_camera = np.zeros_like(pose1)
        new_camera[:3,:3] = inter_rot
        new_camera[:3,3] = inter_pos
        new_camera[3,3] = 1

        new_cameras.append(new_camera)
    return np.stack(new_cameras)

def main(id1, id2, filepath, inter, result):
    with open(filepath, 'r') as fp:
        metab = json.load(fp)
        frames = metab.get("frames")
        # pdb.set_trace()

        for frame in frames:
            id = frame.get("img_id")
            if id == id1:
                pose1 = np.array(frame.get("transform_matrix"))
            if id == id2:
                pose2 = np.array(frame.get("transform_matrix"))
            
        # pdb.set_trace()
        # print(pose1.shape(), pose2.shape())
    re = interpolate_pose_new(pose1, pose2, inter)
    # pdb.set_trace()

    print(re)
    # np.savetxt('output.txt', re ,fmt='%f')
    # np.save(file="data.json", arr=re)
    # json.dumps(re, cls=NumpyEncoder)
    dic={}
    dic['index']=re.tolist()
    dicJson = json.dumps(dic)
    with open(result, 'w') as f:
        json.dump(dicJson, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='To handle which dataset and how to handle it.')

    # Basic arguments
    parser.add_argument("--filepath", type=str, help="json file pathroot")
    parser.add_argument("--id1", type=int, help="image id1")
    parser.add_argument("--id2", type=int, help="image id2")
    parser.add_argument("--inter", type=int, help="inter num")
    parser.add_argument("--result", type=str, help="json file savepath")
    args = parser.parse_args()

    main(args.id1, args.id2, args.filepath, args.inter, args.result)
