import os
import argparse
def write(id, path):
    filelist = os.listdir(path)
    for filename in filelist:
        print('non-makeup/'+filename, ' makeup/'+id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='', help='path to the no_makeup dir')
    parser.add_argument('--path', type=str, default='/data/heyue/makeup_related/SCGAN-master/MT-Dataset/hy/non-makeup/', help='path to the no_makeup dir')
    args = parser.parse_args()
    write(args.id, args.path)