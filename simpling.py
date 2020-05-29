from torch.utils.data  import Dataset
import os
import numpy as np
import torch
from PIL import Image as img


class FaceDataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(path,"positive.txt")).readlines())  #把做好的数据全部读写到dataset中
        self.dataset.extend(open(os.path.join(path,"negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).readlines())

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split(" ")
        image_path = os.path.join(self.path,strs[0])
        cond = torch.Tensor([int(strs[1])])
        offset = torch.Tensor([float(strs[2]),float(strs[3]),float(strs[4]),float(strs[5])])
        img_data = torch.Tensor(np.array(img.open(image_path))/255-0.5)

        print(strs)
        print(img_data.shape)
        a = img_data.permute(2,0,1)#换轴，换成chw
        print(a.shape)

        return a,cond,offset
    def __len__(self):
        return len(self.dataset)



if __name__ == '__main__':
    data = FaceDataset(r"D:\谭\总文件汇总2019\python课件\深度学习课件\深度学习\第十九天20190628\第十九天20190628\视频\celeba_4\12")
    print(data[0])