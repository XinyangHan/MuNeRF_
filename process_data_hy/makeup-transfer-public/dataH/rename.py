import os

path1 = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-publicdata/crop/blend/"
path2 = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-publicdata/crop/after/"
path_new = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-publicdata/crop/blend_color/"
dirlist = os.listdir(path2)
filelist = os.listdir(path1)
#for dir in dirlist:
#    dirname = dir.split('.')[0]
#    newdir = os.path.join(path_new, dirname)
#    os.mkdir(newdir)
#    print('make dir', dirname)
for file in filelist:
    print('filename',file)
    file2 = file[:4]+'.png'
    blend = os.path.join(path1, file)
    blendnew = os.path.join(path_new, file2)
    os.rename(blend, blendnew)
print('rename all blend images and move them to blendcolor')

    
    #before = os.path.join(path2,file)
    #startnum = str(startNumber+count)
    #if not os.path.isfile(blend):
    #    os.remove(before)
    #    continue
    #newF1 = os.path.join(path1,file[:5]+'.png')
    #newF2 = os.path.join(path2,file[:5]+'.png')
    #os.rename(before, newF2)
    #os.rename(after, newF1)
