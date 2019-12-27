import os
path = "/data/qtt/AttnGAN/output/coco_glu-gan2_2018_09_14_13_21_05/Model/netG_epoch_120/valid/single"
files= os.listdir(path)
s = []
for file in files:
     if not os.path.isdir(file):
          f = open(path+"/"+file);
          iter_f = iter(f);
          str = ""
          for line in iter_f:
              str = str + line
          s.append(str)
print(s)
