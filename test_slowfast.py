from mmaction.apis import single_gpu_test,multi_gpu_test
from mmaction.datasets import build_dataloader
from mmcv.parallel import MMDataParallel
from mmaction.apis import inference_recognizer, init_recognizer
from mmcv import Config
from mmcv.runner import set_random_seed
from mmaction.datasets import build_dataset
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import torch
#from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score



#config file and checkpoint
cfg=Config.fromfile('/home/abali/mmaction2/work_dir/my_config.py')
checkpoint='/home/abali/mmaction2/work_dir/slowfast_varuna/epoch_59.pth'
cfg.load_from='/home/abali/mmaction2/work_dir/slowfast_varuna/epoch_59.pth'

#initiate model
model = init_recognizer(cfg, checkpoint, device='cuda:0', use_frames=True)

dataset = build_dataset(cfg.data.test, dict(test_mode=True))
data_loader = build_dataloader(
        dataset,
        videos_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
model = MMDataParallel(model, device_ids=[0])
outputs = single_gpu_test(model, data_loader)

predicted = [np.argmax(out, axis=0) for out in outputs]
dict = {}
for i in range(565):
    if i < 166:
        
        dict[str(i)] = predicted[i] +1
    if i == 166:
        dict[str(i)] = 0
    else:
        dict[str(i)] = predicted[i-1] +1  
print(dict)
import csv
with open('output_test.csv','w') as f:
    w = csv.DictWriter(f,dict.keys())
    w.writeheader()
    w.writerow(dict)
    

df = pd.read_csv('./output_test.csv', header = None)
a = df.T
a.columns = ['id', 'crop_type']
a.to_csv('result.csv', index=False)
