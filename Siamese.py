"""
Siamese network for cylinder matching 

Aim is to match between cylinders and non cylinders so as to generate the predictions

This file will consist of training loop

Assumes that the folder pointed to consist of 0 and 1 subfolders...



"""
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np

class SiamDataset(Dataset):
	"""docstring for SiamDataset"""
	def __init__(self, base, transforms = None):
		super().__init__()
		self.base = base
		self.dir0 = base + '0'
		self.dir1 = base + '1'
		self.combo = self._getlist()
		self.transforms = transforms
		# combo should be a combination of images of same type and different type
		# consist of (path1, path2, 0/1) if same path then 1 else 0

	def _getlist(self):
		
		l = []
		l0 = os.listdir(self.dir0)
		l1 = os.listdir(self.dir1)
		l0 = list(filter(lambda x : x.endswith('.jpg') , l0))
		l1 = list(filter(lambda x : x.endswith('.jpg') , l1))
		l0 = list(map(lambda x : self.dir0 + '/' + x , l0))
		l1 = list(map(lambda x : self.dir1 + '/' + x , l1))

		# generate all combinations 
		tmp = []

		for i in range(len(l0)):
			for j in range(i+1, len(l0)):
				tmp.append( (l0[i] , l0[j] , 0 ) )

		for i in range(len(l1)):
			for j in range(i+1, len(l1)):
				tmp.append( (l1[i] , l1[j] , 0 ) )

		for i in range(len(l0)):
			for j in range(len(l1)):
				tmp.append( (l0[i] , l1[j] , 1 ) )


		assert len(tmp) == len(l0)*len(l1) + (len(l0)*(len(l0)-1))//2 + (len(l1)*(len(l1)-1))//2
		return tmp


	def image2tensor(self , path):

		# given a path first read the image
		img = cv2.imread(path)
		img = cv2.resize(img, (100,100))
		# img = cv2.cvtColor( img , cv2.COLOR_BGR2GRAY )
		img = img[:,:,::-1].copy()
		img = img.transpose( ( 2,0,1 ) )
		img_tensor = torch.from_numpy(img).float().div(255) # normalise the image
		return img_tensor


	def tensor2image(self, tensor):
		img = tensor.numpy().squeeze(0)
		img = img.transpose((1,2,0))
		img = img[:,:,::-1].copy()
		img = img * 255# assuming a normalized tensor
		img = img.astype('uint8')
		return img

	# def __call__(self):
	# 	pass

	def __getitem__(self , idx):
		# need to return the item as an image file
		z = self.combo[idx]
		return ( self.image2tensor(z[0]), self.image2tensor(z[1]), torch.from_numpy( np.array(z[2] , dtype=np.float32)  ) )


	def __len__(self):
		return len(self.combo)
		# defining this is a must else the network doesn't know the size

	def __str__(self):
		return self.dir0 + ' 0 and 1 ' + self.dir1




if __name__ == '__main__':
	
	data = SiamDataset('./utils/')
	# print(data.combo)
	dataloader = DataLoader(data, batch_size=1)

	global ii
	it = iter(dataloader)
	z = it.next()
	print(len(z))
	print(z[1].size())
	im = data.tensor2image(z[1])

	# print( (ii  == im).all() )
	# why unable to print wehn all same
	# because int type is scaled inside opencv
	while 1:

		cv2.imshow('sd',im)
		key = cv2.waitKey() & 0xFF
		if key == ord('q'):
			break

	cv2.destroyAllWindows()
	# print(z[0].size())


