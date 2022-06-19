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
'''
def _name_to_int(name):

    integer=0
    if name=='clap':
        integer=1
    elif name=='flapping':
        integer=2
    elif name=='jump-up':
        integer=3
    elif name=='to-taste':
        integer=4
    elif name=='others':
        integer=5
    elif name=='arm-indeterminate-movement':
        integer=6
    elif name=='ball':
        integer=7
    elif name=='clinician-breath':
        integer=8
    elif name=='clinician-says-look':
        integer=9
    elif name=='sensorial-hand':
        integer=10
    elif name=='sensorial-mouth':
        integer=11
    elif name=='stereotypy':
        integer=12
    elif name=='anniversary':
        integer=13
    elif name=='bubbles':
        integer=14

   # elif name=='sway':
   #     integer=4
    return integer

def make_label(test_file):
    label = []
    with open(test_file, "r") as f:
         data = [i.strip() for i in f.readlines()]
    i = 0
    for vid in data:
         label.append((_name_to_int(vid.split('_')[0]))-1)

#    print(label)
    return label
'''
'''targets = make_label('/home/abali/mmaction2/work_dir/splits_activis/new_test.txt')'''


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

'''
#print(predicted)

f1_micro = f1_score(targets, predicted, average='micro')
f1_macro = f1_score(targets, predicted, average='macro')
f1_weighted = f1_score(targets, predicted, average='weighted')

print('F1 score: micro: ', f1_micro, 'macro: ', f1_macro, 'Weighted: ', f1_weighted)


CMat = dataset.confuse_matrix(outputs)
CMat = CMat.astype('float')/ CMat.sum(axis=1)[:, np.newaxis]
# save numpy array as npy file
from numpy import asarray
from numpy import save

target_names = ['clap', 'arm-flapping', 'jump-up', 'to-taste', 'others', 
'arm-indeterminate-movement', 'ball', 'clinician-breath', 'clinician-says-look', 'sensorial-hand'
,'sensorial-mouth', 'stereotypy', 'anniversary', 'bubbles']
#save('/home/abali/mmaction2/work_dir/results/x3d_act_5_epo30_cmat.npy', CMat)
import seaborn as sn
plt.figure(figsize = (30,30))
ax= plt.subplot()
sn.set(font_scale=1.3) # for label size
sn.heatmap(CMat, cmap='Blues',annot=True, fmt='0.2f', xticklabels=target_names, yticklabels=target_names,  annot_kws={"size": 10})#,ax=ax) # font size
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')

#plt.savefig('/home/abali/mmaction2/work_dir/results/x3d_act_5_epoch30.png')
plt.show()
plt.close()
print('done cmat')
'''
'''
eval_config = cfg.evaluation
eval_config.pop('interval')
eval_res = dataset.evaluate(outputs, **eval_config)
for name, val in eval_res.items():
    print(f'{name}: {val:.04f}')

'''
