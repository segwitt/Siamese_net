
from torch.utils.data import Dataset , DataLoader
import torchvision
from torchvision import transforms
import pandas as pd
import os
from pathlib import Path
import numpy as np

def preprocess(img=None, path=None):
    import cv2
    import matplotlib as mpl
    mpl.use('TkAgg')  # or whatever other backend that you want
    import matplotlib.pyplot as plt
    # from matplotlib import pyplot as plt
    # image is a numpy array from cv2.imread
    if path is not None:
        img = cv2.imread(path)
    img = img[:,:,-1]
    plt.imshow(img)
    plt.show()
    # img = img.transpose((2,1,0))
    # print(img.shape)



class CustomDataset(Dataset):
    
    def __init__(self, dirpath='./', transform = None):
        super().__init__()
        self.transform = transform
        self.dirpath = dirpath
        self.id2name, self.name2id = self._getFolderNamesAndIds()
        self.data = self._getPairsAndValue( self.id2name )



    def __getitem__(self, idx):
        # return the image vector/np_array by reading the image from the file path
        import cv2
        x, y, xid, yid = self.data[idx]
        x_path = os.path.join(self.id2name[xid], x)
        y_path = os.path.join(self.id2name[yid], y)

        return x_path, y_path

        img1 = cv2.imread(x_path)
        img2 = cv2.imread(y_path)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return x_path, y_path, int(xid==yid)
   

    def __len__(self):
        return len(self.data)
    

    def _getFolderNamesAndIds(self):
        # expects multiple folders at basedir/training, basedir/testing
        basedir = self.dirpath
        data_folder = Path(basedir)
        training_folder = data_folder / "training"
        testing_folder = data_folder / "testing"
        foldernames = os.listdir(training_folder)
        # these many folders need to be processed
        id2name, name2id = {}, {}
        isdir = lambda path : os.path.isdir(path)
        for idx, foldername in enumerate(foldernames):
            base_path = training_folder / foldername
            # print(abs_path)
            if isdir(base_path):
                id2name[idx] = str(base_path)
                name2id[(base_path)] = idx
        
        return id2name, name2id
    

    def _getPairsAndValue(self, id2name):
        # make pairs of the file names
        # id2names is a directory path and we need to list all the images in that path
        list_images = [ [ ( x , y ) for y in os.listdir(id2name[x]) ] for x in id2name ]
        
        for idx, l in enumerate(list_images):
            list_images[idx] = list_images[idx][:5]
        
        ls = []
        for i in range(len(list_images)):
            for j in range(len(list_images)):
                for k in range(5):
                    for l in range(5):
                        if j < i or l < k:
                            continue
                        x = list_images[i][k][1]
                        y = list_images[j][l][1]
                        xid = list_images[i][k][0]
                        yid = list_images[j][l][0]
                        ls.append( ( x, y, xid, yid ) )
        # print(len(ls))
        return ls
        # print(list_images[0])





def main():
    dset = CustomDataset(dirpath='./faces')
    print(dset[1])
    preprocess(path=dset[1][0])
    # x = dset._getFolderNamesAndIds()
    # dset._getPairsAndValue(*x)
    # d_f = Path('./faces') / 'training'
    # dd_f = d_f / 's1'
    # isdir = lambda x : os.path.isdir(x)
    # print(str(dd_f))



if __name__ == '__main__':
    main()