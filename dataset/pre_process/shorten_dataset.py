import os
import json
import pdb

targets = ['/data/hanxinyang/MuNeRF_latest/dataset/girl10_test']


for target in targets:
    basedir = target
    splits = ["train", "val", "test"]
    for split in splits:
        metas = {}
        for s in splits:
            with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
                # a = fp.read()
                metas[s] = json.load(fp)
                # a = json.loads(a)

                raw = metas[s]
                new = {}                
                new['focal_len'] =  metas[s]['focal_len']
                new['cx'] = metas[s]['cx']
                new['cy'] = metas[s]['cy']
                new_frames = []
                for i in range(len(metas[s]['frames'])):
                    if metas[s]['frames'][i]['img_id'] < 1800:
                        new_frames.append(metas[s]['frames'][i])
                new['frames'] = new_frames

                new_json_path = os.path.join(basedir, f"new_transforms_{s}.json")
                with open(new_json_path, 'w') as f:
                     json.dump(new, f,ensure_ascii=False)
                # with open(new_json_path, 'r') as t:
                #     content = json.load(t)
                # pdb.set_trace()
                
                json_path = os.path.join(basedir, f"transforms_{s}.json")
                json_path_ori = os.path.join(basedir, f"ori_transforms_{s}.json")
                os.system(f"mv {json_path} {json_path_ori}")
                os.system(f"mv {new_json_path} {json_path}")
            
                
                
        with open(os.path.join(basedir, 'box.json'), "r") as fp:
            metab = json.load(fp)
            
            
