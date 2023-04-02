import pickle
import os

def write_TestdataOrTraindata(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print('data', len(data.keys()))
    #f=open(filename[filename.find('_t')+1:filename.find('.')]+'.txt','w',encoding='utf-8')
    #for i in range(len(data)):
    #    print(data[i])
    #    f.write(str(data[i]))
    #    f.write('\n')
 
if __name__=='__main__':
    #test_filename,train_filename=getFilename()
    test_filename = '/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-publicdata/crop/landmark.pk'
    write_TestdataOrTraindata(test_filename)
