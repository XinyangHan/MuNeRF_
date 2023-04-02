import json
jsonfile = 'D:\\hy_goo\\dave_dvp\\new\\transforms_train.json'
jsonfile_train = 'D:\\hy_goo\\dave_dvp\\new\\train.json'
jsonfile_test = 'D:\\hy_goo\\dave_dvp\\new\\test.json'
#old_dict = json.loads(jsonfile)
with open(jsonfile,'r') as load_f:
    old_dict = json.load(load_f)
new_dict_test = {}
new_dict_test['camera_angle_x'] = old_dict['camera_angle_x']
new_dict_test['frames'] = old_dict['frames'][:200]
new_dict_test['intrinsics'] = old_dict['intrinsics']
new_dict_train = {}
new_dict_train['camera_angle_x'] = old_dict['camera_angle_x']
new_dict_train['frames'] = old_dict['frames'][200:1000]
new_dict_train['intrinsics'] = old_dict['intrinsics']

with open(jsonfile_test,"w") as f:
    json.dump(new_dict_test,f,indent=1)

with open(jsonfile_train,"w") as f:
    json.dump(new_dict_train,f,indent=1)