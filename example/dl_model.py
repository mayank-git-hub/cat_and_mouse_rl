import sys
from torchvision import datasets, transforms
import torchvision
import torch.utils.data as data
import torch
from torch.autograd import Variable
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
# Create your models here.
class dl_model():

	def __init__(self, data_loader, model, use_cuda = True, train_transform = None, target_transform = None, test_transform = None,
	 PreTrained=True, description='Machine Learning Model', version='1.0', Category='Machine Learning', SubCategory='Model',
	  dict_hp = {}, train_dir='datasets/Train', test_dir='datasets/Test', target_train_dir = 'datasets/Train_anno', target_test_dir = 'datasets/Test_anno',cpu_alloc=[4, 1], data_type='',logger_dir=''):
		if train_transform == None:

			self.train_transform = transforms.Compose([
											# transforms.RandomHorizontalFlip(),
											# torchvision.transforms.RandomVerticalFlip(),
											#torchvision.transforms.RandomCrop(size = (100, 100)),
											torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
											# torchvision.transforms.RandomRotation(10, resample=False, expand=False, center=None),
											transforms.ToTensor(),
											])
		else:
			
			self.train_transform = train_transform

		if test_transform == None:

			self.test_transform = transforms.Compose([
											 #transforms.CenterCrop((331, 331)),
											 transforms.ToTensor(),
											 ])
		else:

			self.test_transform = test_transform
		
		if target_transform == None:

			self.target_transform = transforms.Compose([
											 #transforms.CenterCrop((331, 331)),
											 transforms.ToTensor(),
											 ])
		else:
			self.target_transform = None

		self.factor = 1.5
		self.description = description
		self.version = version
		self.Category = Category
		self.SubCategory = SubCategory
		self.hp = dict_hp
		self.PreTrained = PreTrained
		
		self.train_dir = train_dir
		self.test_dir = test_dir
		
		self.train_data = data_loader(root=train_dir ,Type = 'Train', transform=self.train_transform, target_transform = self.target_transform, target_file_add = target_train_dir, factor=self.factor)
		self.test_data = data_loader(root=test_dir ,Type = 'Test', transform=self.test_transform, target_transform = self.target_transform, target_file_add = target_test_dir, factor=self.factor)

		self.train_data_loader = data.DataLoader(self.train_data, batch_size=self.hp['train_batch_size'], shuffle=True, num_workers=cpu_alloc[0])
		self.test_data_loader = data.DataLoader(self.test_data, batch_size=self.hp['test_batch_size'], shuffle=False, num_workers=cpu_alloc[1])
		# fusion = FusionGenerator(3,1,32,0.).cuda()
		self.model = model(n_channels=3, n_classes=1, lr=self.hp['lr'], opt=self.hp['optimiser'], lossf=self.hp['loss'], logger_dir='', PreTrained=self.PreTrained, on_cuda = use_cuda)
		#  
		self.training_info = {'Loss': [], 'Acc': [], 'Keep_log': True, 'Count':0}
		self.testing_info = {'Loss': 0, 'Acc': 0, 'Keep_log': False, 'Count':0}
		
		self.model_best = {'Loss': sys.float_info.max, 'Acc': 0.0}

		self.epoch_start = 1
		self.start_no = 0

		if use_cuda:
			self.use_cuda = True
			self.model.cuda()
		else:
			self.use_cuda = False

		if self.PreTrained == True:
			self.epoch_start, self.training_info = self.model.load('3750_8_checkpoint.pth.tar', '3750_8_info_checkpoint.pth.tar')
			self.start_no = 0
			self.epoch_start = 2
			print('Loaded the model')

		self.threshold = 0.5



	def __str__(self):

		list_of_things_to_print = [self.description, self.version, self.Category, self.SubCategory, self.hp, self.train_dir, self.model]
		name_of_things = ['Description', 'Version', 'Category', 'SubCategory', 'Hyper Parameters', 'Training Directory', 'Model Architecture']

		string_out = ''

		for name, print_ in zip(name_of_things, list_of_things_to_print):
			string_out += name+' = \n\n'+print_+'\n\n'

		return string_out


	def train_model(self):

		try:

			self.train_data.get_all_names_refresh()
			self.test_data.get_all_names_refresh()

			self.model.requires_grad = True

			self.model.train()

			self.model.opt.zero_grad()

			for epoch_i in range(self.epoch_start, self.hp['epoch']+1):

				for no, (file_name, data, target) in enumerate(self.train_data_loader):

					# plt.imshow(data[0].numpy().transpose(1, 2, 0))
					# plt.show()

					# plt.imshow(target[0].squeeze().numpy())
					# plt.show()

					data, target = Variable(data), Variable(target)

					# plt.imshow(data[0].data.cpu().numpy().transpose(1, 2, 0))
					# plt.show()

					# plt.imshow(target[0].data.cpu().numpy().transpose(1, 2, 0)[:, :, 0])
					# plt.show()

					if self.use_cuda:
						data, target = data.cuda(), target.cuda()

					data = self.model(data)

					# temp = data[0].data.cpu().numpy().transpose(1, 2, 0)

					# plt.imshow(temp[:, :, 0])
					# plt.show()

					# plt.imshow(temp[:, :, 1])
					# plt.show()

					loss = self.model.loss(data, target, self.training_info)

					loss.backward()

					if (self.start_no + no)%self.hp['cummulative_batch_steps']==0:

						self.model.opt.step()

						self.model.opt.zero_grad()

					if (self.start_no + no)%self.hp['log_interval_steps'] == 0 and (self.start_no + no) != 0:

						self.model.save(no=(self.start_no + no), epoch_i=epoch_i, info = self.training_info)

					if (self.start_no + no)%self.hp['print_log_steps'] == 0 and (self.start_no + no) != 0:

						self.model.print_info(self.training_info)
						print()

					if (self.start_no + no)%self.hp['test_now'] == 0 and (self.start_no + no)!=0:

						self.test_model()

						self.model.train()
						self.model.requires_grad = True
						self.model.opt.zero_grad()
					if (self.start_no + no) == len(self.train_data_loader) - 1:
						break

				self.model.save(no=0, epoch_i=epoch_i, info = self.training_info)

				self.training_info = {'Loss': [], 'Acc': [], 'Keep_log': True, 'Count':0}

				self.start_no = 0

			return True

		except KeyboardInterrupt:

			return False

	def test_one_model(self, path):

		print('Testing one model')

		self.model.requires_grad = False

		self.model.eval()

		data = Image.fromarray(plt.imread(path), 'RGB')

		data, _ = self.train_data.aspect_resize(data, 512)

		# image = image.transpose(2, 0, 1)

		# image = torch.from_numpy(np.array(image)[None])

		data = self.train_data.transform(data).unsqueeze(0)

		data = Variable(data)

		if self.use_cuda:
			data = data.cuda()

		data_out = self.model(data)

		data = (data.transpose(1, 3).transpose(1, 2).data.cpu().numpy()*255).astype(np.uint8)

		data_out = data_out.transpose(1, 3).transpose(1, 2).data.cpu().numpy()

		center = np.zeros([data_out.shape[0], data_out.shape[1], data_out.shape[2]])

		center = ((data_out[:, :, :, 0]>self.threshold).astype(np.float32)*255).astype(np.uint8)

		image, contours, hierarchy =  cv2.findContours(center[0] ,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

		center_co = []
		cont = []
		full_cont = []
		# print(np.array(contours).shape)

		for co_ in contours:

			left, right = np.min(co_[:, 0, 0]), np.max(co_[:, 0, 0])
			up, down = np.min(co_[:, 0, 1]), np.max(co_[:, 0, 1])
			center_co.append([up, down, left, right, (up + down)//2, (right + left)//2])
			cont.append(np.array([[left, up], [right, up], [right, down], [left, down]]).reshape([4, 1, 2]))
			left_con, right_con = (right + left)//2 - int((right - left)/2*self.factor), (right + left)//2 + int((right - left)/2*self.factor)
			up_con, down_con = (up + down)//2 - int((down - up)/2*self.factor), int((up + down)//2 + (down - up)/2*self.factor)
			full_cont.append(np.array([[left_con, up_con], [right_con, up_con], [right_con, down_con], [left_con, down_con]]).reshape([4, 1, 2]))


		# print(np.array(cont).shape)

		test = cv2.drawContours(data[0] ,full_cont,-1,(0,255,0),1)

		print(test.shape)

		plt.imsave('test.png', test)



	def test_model(self):

		print('Testing Mode')

		try:

			self.model.requires_grad = False

			self.model.eval()

			for no, (file_name, data, target) in enumerate(self.test_data_loader):

				data, target = Variable(data), Variable(target)

				if self.use_cuda:
					data, target = data.cuda(), target.cuda()

				data_out = self.model(data)

				if not os.path.exists('../Temporary'):
					os.mkdir('../Temporary')

				loss = self.model.loss(data_out, target, self.testing_info)

				data = (data.transpose(1, 3).transpose(1, 2).data.cpu().numpy()*255).astype(np.uint8)

				data_out = data_out.transpose(1, 3).transpose(1, 2).data.cpu().numpy()

				target = target.data.cpu().numpy()

				center = np.zeros([data_out.shape[0], data_out.shape[1], data_out.shape[2]])

				center = ((data_out[:, :, :, 0]>self.threshold).astype(np.float32)*255).astype(np.uint8)

				for no, center_i in enumerate(center):

					image, contours, hierarchy =   cv2.findContours(center_i ,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

					center_co = []
					cont = []
					full_cont = []
					# print(np.array(contours).shape)

					for co_ in contours:

						left, right = np.min(co_[:, 0, 0]), np.max(co_[:, 0, 0])
						up, down = np.min(co_[:, 0, 1]), np.max(co_[:, 0, 1])
						center_co.append([up, down, left, right, (up + down)//2, (right + left)//2])
						cont.append(np.array([[left, up], [right, up], [right, down], [left, down]]).reshape([4, 1, 2]))
						left_con, right_con = (right + left)//2 - int((right - left)/2*self.factor), (right + left)//2 + int((right - left)/2*self.factor)
						up_con, down_con = (up + down)//2 - int((down - up)/2*self.factor), int((up + down)//2 + (down - up)/2*self.factor)
						full_cont.append(np.array([[left_con, up_con], [right_con, up_con], [right_con, down_con], [left_con, down_con]]).reshape([4, 1, 2]))


					# print(np.array(cont).shape)
					# print(data_out[no][:, :, 0].shape)

					# plt.imshow(data_out[no][:, :, 0])

					plt.imsave('../Temporary/'+file_name[no][:-3]+'.png', cv2.drawContours(data[no] ,full_cont,-1,(0,255,0),1))

					# plt.show()

					# plt.imshow(data[no])

					# plt.show()

					# plt.imshow(target)


			print('Testing Completed successfully')
			print('Test Results\n\n', self.testing_info)

			if self.testing_info['Acc'] > self.model_best['Acc']:

				print('Found a new best model')

				self.model_best['Acc'] = self.testing_info['Acc']
				
				self.model.save(no=0, epoch_i=0, info = self.testing_info)

			self.testing_info = {'Loss': 0, 'Acc': 0, 'Keep_log': False, 'Count':0}

			return True

		except KeyboardInterrupt:

			print('Testing Interrupted')

			return False
