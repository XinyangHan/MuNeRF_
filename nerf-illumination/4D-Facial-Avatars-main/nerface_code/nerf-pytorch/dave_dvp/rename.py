import os

i=1

trainpath = 'D:\\hy_goo\\dave_dvp\\new\\train'
for d in os.listdir(trainpath):
    n = len(str(i))
    numstr = 'f_'+'0'*(4-n)+str(i)
    filename = numstr+'.png'
    os.rename(os.path.join(trainpath, d), os.path.join(trainpath, filename))
    i=i+1