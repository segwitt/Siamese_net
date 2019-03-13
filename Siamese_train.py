import Siamese
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

global base_img

class Net(nn.Module):
	"""docstring for Net"""
	def __init__(self):
		super().__init__()

		self.cnn1 = nn.Sequential(
			nn.ReflectionPad2d(1),
			nn.Conv2d(3, 4, kernel_size=3),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(4),
			nn.Dropout2d(p=.2),
			
			nn.ReflectionPad2d(1),
			nn.Conv2d(4, 8, kernel_size=3),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(8),
			nn.Dropout2d(p=.2),

			nn.ReflectionPad2d(1),
			nn.Conv2d(8, 8, kernel_size=3),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(8),
			nn.Dropout2d(p=.2),

			nn.ReflectionPad2d(1),
			nn.Conv2d(8, 8, kernel_size=3),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(8),
			nn.Dropout2d(p=.2),
		)

		self.fc1 = nn.Sequential(
			nn.Linear(8*100*100, 500),
			nn.ReLU(inplace=True),

			nn.Linear(500, 500),
			nn.ReLU(inplace=True),

			nn.Linear(500, 5)
		)

	def forward_once(self, x):
		output = self.cnn1(x)
		output = output.view(output.size()[0], -1)
		output = self.fc1(output)
		return output
	def forward(self, input1, input2):
		output1 = self.forward_once(input1)
		output2 = self.forward_once(input2)
		return output1 , output2


class ContrastiveLoss(torch.nn.Module):
	"""
	Contrastive loss function.
	Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	"""

	def __init__(self, margin=6.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin

	def forward(self, output1, output2, label):
		euclidean_distance = F.pairwise_distance(output1, output2)
		loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
									  (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

		return loss_contrastive


def train(epochs=10):

	net = Net()
	criterion = ContrastiveLoss()
	optimizer = optim.Adam(net.parameters(), lr=0.0005)


	counter = []
	loss_history = []
	iter_num = 0
	epochs = epochs


	data = Siamese.SiamDataset('./utils/')

	dloader = DataLoader(data, batch_size=3, shuffle=True)



	for epoch in range(epochs):

		for i, data in enumerate(dloader):
			# print(len(data), data[0].size() , data[2])
			# print()
			img0, img1, label = data
			output1, output2 = net(img0, img1)
			optimizer.zero_grad()
			loss_contrastive = criterion(output1, output2, label)
			loss_contrastive.backward()
			optimizer.step()
			if i%20==0:
				print("epc num {} curr loss {} ".format(epoch, loss_contrastive.item()))
				iter_num += 2
				counter.append(iter_num)
				loss_history.append(loss_contrastive.item())

	torch.save(net, 'model.pth')

	import matplotlib as mpl
	mpl.use('TkAgg')  # or whatever other backend that you want, (maybe disable on windows)
	import matplotlib.pyplot as plt
	plt.plot(counter, loss_history)
	# plt.plot(test_losses, label='test_losses')
	# plt.legend(frameon=False)
	plt.show()


def imshow(img,text=None,should_save=False):
	import matplotlib as mpl
	mpl.use('TkAgg')  # or whatever other backend that you want, (maybe disable on windows)
	import matplotlib.pyplot as plt
	npimg = img.numpy()
	plt.axis("off")
	if text:
		plt.text(75, 8, text, style='italic',fontweight='bold',
			bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

def test(path, tots=10):
	# paht to load the entire model
	import torchvision
	model = torch.load(path)
	model.eval()
	print(model)

	test_data = Siamese.SiamDataset('./utils/')
	test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
	print('lenn',len(test_loader))
	it = iter(test_loader)
	x0,_,_ = it.next()
	print(x0.size())

	for i in range(tots):
		_,x1,label2 = next(it)
		concatenated = torch.cat((x0,x1),0)
		
		output1,output2 = model(x0 , x1)
		euclidean_distance = F.pairwise_distance(output1, output2)
		imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))


def video_test(path_to_vid, path_to_model):

	import cv2
	import time

	def im_to_tensor(img, resize=None, dim0=False):
		# img is an opencv numpy image
		if resize is not None:
			img = cv2.resize(img, resize)

		img = img[:,:,::-1].copy()
		img = img.transpose( ( 2,0,1 ) )
		img_tensor = torch.from_numpy(img).float().div(255) # normalise the image
		if dim0:
			img_tensor = img_tensor.view(1, *img_tensor.size())
		return img_tensor

	global base_img
	print('base_img_path ', base_img)
	fixed_img_tensor = im_to_tensor( cv2.imread(base_img) , resize=(100,100), dim0=True)

	# return 

	cam = cv2.VideoCapture(path_to_vid)
	cnt = 0
	prev = None

	x,y = 600,100
	h,w = 150,150
	cnt=0
	cnt2=0
	# img = torch.randn(1, 3, 100 ,100)
	# model loaded from file
	model = torch.load(path_to_model)

	frame_rate = 100//3

	cv2.namedWindow('frame', 0)
	cv2.resizeWindow('frame' , (800,800))

	# cv2.namedWindow('frame0', 0)
	# cv2.resizeWindow('frame0' , (800,800))
	pre = 1
	timer = 0
	while True:

		tick = time.time()
		ret, frame = cam.read()
		if not ret: break
		frame0 = frame.copy()
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
		cv2.line(frame, (x,y+35), (x+100,y+35), (255,0,0), 2)
		# cv2.rectangle(frame, (x-30, y), (x+w-30, y+h), (255,0,0), 2)
		# cv2.rectangle(frame, (x+30, y), (x+w+30, y+h), (255,0,0), 2)
		cv2.imshow('frame', frame)
		# cv2.imshow('frame0', frame0)
		key = cv2.waitKey(frame_rate) & 0xFF
		frame2 = frame0[y:y+h, x:x+w,:]
		# frame3 = frame0[y:y+h, x-30:x+w-30,:]
		# frame4 = frame0[y:y+h, x+30:x+w+30,:]
		out1, out2 = model( fixed_img_tensor, im_to_tensor(frame2, resize=(100,100), dim0=True) )
		loss = F.pairwise_distance(out1, out2)
		loss_val = loss.item()
		print('loss ', loss.item())

		if loss_val < .5:
			if pre == 0 and timer >= 2:
				cnt+=1
				timer = 0
			pre =  1
		else :
			pre  = 0

		print(cnt)

		# if key == ord('m'):
		# 	cv2.imwrite('./1/'+str(cnt)+'.jpg', frame2)
		# 	cnt+=1
		# 	# cv2.imwrite('./1/'+str(cnt)+'.jpg', frame3)
		# 	# cnt+=1
		# 	# cv2.imwrite('./1/'+str(cnt)+'.jpg', frame4)
		# 	# cnt+=1
		# elif key == ord('c'):
		# 	cv2.imwrite('./0/'+str(cnt2)+'.jpg', frame2)
		# 	cnt2+=1
		# 	# cv2.imwrite('./0/'+str(cnt2)+'.jpg', frame3)
		# 	# cnt2+=1
		# 	# cv2.imwrite('./0/'+str(cnt2)+'.jpg', frame4)
		# 	# cnt2+=1
		if key == ord('q'):
			break
		tock = time.time()
		timer += (tock - tick)
		print( (tock - tick) * 1000)

	cam.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	
	global base_img
	base_img = './utils/1/2.jpg'
	video_test('./utils/cyl2.mp4', './model.pth')


	# train
	#save model.pth to local directory
	# train(epochs=20)

	# test
	# test('model.pth')