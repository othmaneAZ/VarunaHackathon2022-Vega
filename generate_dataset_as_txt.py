import os
import numpy as np
def generate_txt(dir_path, output_txt):
   f = open(output_txt, "w")
   for element in os.listdir(dir_path):
       element_0 = element
       #element = element.split("_")
       label = element
       element_path = dir_path + '/' + element_0
       num_frames = len(os.listdir(element_path))
       f.write(element_path + ' ' + str(num_frames) + ' ' + '0' + '\n') 
      '''
      we dont have a lebel, so we will use a dummy label aka 0 to make the file compatible to mmaction input.
      '''
    
generate_txt('/data/stars/user/mguermal/varuna/frames_varuna_test','/data/stars/user/mguermal/varuna/anno_test_file.txt')
