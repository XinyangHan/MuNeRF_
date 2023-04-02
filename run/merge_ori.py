import os

sources = ['train', 'test', 'val']
target = '/home/yuanyujie/cvpr23/dataset/girl8/ori_imgs'
os.makedirs(target, exist_ok=True)
for source in sources:
    source_path = os.path.join('/home/yuanyujie/cvpr23/dataset/girl8', source)
    for thing in os.listdir(source_path):
        thing_path = os.path.join(source_path, thing)
        target_path = os.path.join(target, thing)
        os.system(f"cp {thing_path} {target_path}")