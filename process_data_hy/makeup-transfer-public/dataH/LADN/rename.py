import os

path2 = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-publicdata/LADN/after2"
#path2 = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-publicdata/LADN/before"
filelist = os.listdir(path2)
for file in filelist:
    print('filename',file)
    #file2 = file[-9:-4]+'_'+file[:5]+'.jpg'
    file2 = file[:5]+'.jpg'
    blend = os.path.join(path2, file)
    blend2 = os.path.join(path2, file2)
    os.rename(blend, blend2)