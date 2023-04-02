import torch
import psutil
import time

pid = 1900
gpu_id = 3

while psutil.pid_exists(pid):
    print('\r PID: ' + str(pid) + ' exists. Waiting ...', end='')
    time.sleep(1)

print('\nTrain Time ...')
a = torch.zeros([1000*1000*50*5], dtype=torch.float64).cuda(gpu_id)
while True:
    b = torch.sin(torch.exp(a)) + torch.cos(torch.exp(a))
