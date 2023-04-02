import os
import json
import argparse

def filter(input, output):
    filelist1 = sorted(os.listdir(input)) #whole
    #filelist2 = sorted(os.listdir(refer)) #refer
    #filelist2 = sorted(os.listdir(output)) #output
    for i in range(len(filelist1)):
        filename = filelist1[i]
        newname = filename[:4]+'.png'
        os.rename(os.path.join(input, filelist1[i]), os.path.join(output, newname))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='', help='path to the no_makeup dir')
    #parser.add_argument('--refer', type=str, default='', help='path to the result dir')
    parser.add_argument('--output', type=str, default='', help='path to the result dir')
    args = parser.parse_args()
    inputpath = args.input
    #referpath = args.refer
    outputpath = args.output
    filter(inputpath, outputpath)
    os.rmdir(inputpath)
