import numpy as np

npyfile = 'D:\\hy_goo\\dave_dvp\\new\\index_map.npy'
npynewfile = 'D:\\hy_goo\\dave_dvp\\new\\index.npy'
npy = np.load(npynewfile)
#npynew = npy[200:1000]
print('npyfile',npy,len(npy))
#np.save(npynewfile,npynew)