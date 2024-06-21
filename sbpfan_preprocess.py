import numpy as np
import mne
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

AZ_Data = []

path='/content/Ayan Sir Server/Alz_Dem_HC/Dataset/0'
files = glob.glob(path + "/*.set")
for filename in files:
  data = mne.io.read_raw(filename)
  raw_data = data.get_data()
  AZ_Data.append(raw_data)


normalise_AZ_data=[]

for i in  range(len(AZ_Data)):
  std=np.std(AZ_Data[i],axis=1)
  mean=np.mean(AZ_Data[i],axis=1)
  AZ_Data[i]=(AZ_Data[i].transpose()-mean.transpose()).transpose()
  AZ_Data[i]=(AZ_Data[i].transpose()/std.transpose()).transpose()
  normalise_AZ_data.append(AZ_Data[i])

winSize=500*8 # Size of data point (data of 8 sec)
stride=500*6  # Sliding window with stride 6 sec for 2 sec overlap (25%)
count=0

windowing_AZ_data=[]

for i in range(0,np.shape(normalise_AZ_data[35])[1]-winSize,stride):
  count+=1
  if len(np.shape(windowing_AZ_data))>1:
    windowing_AZ_data=np.dstack((windowing_AZ_data,normalise_AZ_data[35][:,i:i+winSize]))
  else:
    windowing_AZ_data=np.reshape(normalise_AZ_data[35][:,i:i+winSize],(19,np.shape(normalise_AZ_data[35][:,i:i+winSize])[1],1))

print(count)

np.array(windowing_AZ_data).shape

w = 3
h = 2
d = 100
count = windowing_AZ_data.shape[2]
path = '/content/Ayan Sir Server/Alz_Dem_HC/Generated Images/0/'
plt.figure(figsize=(w, h), dpi=d)
cmap = plt.get_cmap('inferno')

for i in tqdm(range(count), desc="Generating Images"):
    plt.pcolormesh(windowing_AZ_data[:, :, i], cmap=cmap)
    plt.axis('off')
    plt.savefig(path + 'AZ-FD-HC-36-' + str(i) + '.png', bbox_inches='tight', pad_inches=0)

plt.close()