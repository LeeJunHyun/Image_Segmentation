import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net


class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.BCELoss()

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=3,output_ch=1)
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(img_ch=3,output_ch=1,t=self.t)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch=3,output_ch=1)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch=3,output_ch=1,t=self.t)
			

		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)

		self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img


	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		
		unet_path = os.path.join(self.model_path, '%s-%d.pkl' %(self.model_type,self.num_epochs))

		# U-Net Train
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		else:
			# Train for Encoder
			lr = self.lr
			best_unet_acc = 0.
			
			for epoch in range(self.num_epochs):

				self.unet.train(True)
				epoch_loss = 0
				
				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length = 0

				for i, (images, GT) in enumerate(self.train_loader):
					# GT : Ground Truth

					images = images.to(self.device)
					GT = GT.to(self.device)

					# SR : Segmentation Result
					SR = self.unet(images)
					SR_probs = F.sigmoid(SR)
					SR_flat = SR_probs.view(SR_probs.size(0),-1)

					GT_flat = GT.view(GT.size(0),-1)
					loss = self.criterion(SR_flat,GT_flat)
					epoch_loss += loss.item()

					# Backprop + optimize
					self.reset_grad()
					loss.backward()
					self.optimizer.step()

					acc += get_accuracy(SR,GT)
					SE += get_sensitivity(SR,GT)
					SP += get_specificity(SR,GT)
					PC += get_precision(SR,GT)
					F1 += get_F1(SR,GT)
					JS += get_JS(SR,GT)
					DC += get_DC(SR,GT)
					length += images.size(0)

				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length

				# Print the log info
				print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
					  epoch+1, self.num_epochs, \
					  epoch_loss,\
					  acc,SE,SP,PC,F1,JS,DC))

				if (epoch+1) % self.log_step ==0:
					torchvision.utils.save_image(images.data.cpu(),
													os.path.join(self.result_path,
																'%s_train_%d_image.png'%(self.model_type,epoch+1)))

					torchvision.utils.save_image(SR.data.cpu(),
													os.path.join(self.result_path,
																'%s_train_%d_SR.png'%(self.model_type,epoch+1)))

					torchvision.utils.save_image(GT.data.cpu(),
													os.path.join(self.result_path,
																'%s_train_%d_GT.png'%(self.model_type,epoch+1)))



				# Decay learning rate
				if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
					lr -= (self.lr / float(self.num_epochs_decay))
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print ('Decay learning rate to lr: {}.'.format(lr))
				
				
				#===================================== Validation ====================================#
				self.unet.train(False)
				self.unet.eval()

				if (epoch+1) % self.val_step ==0:
					acc = 0.	# Accuracy
					SE = 0.		# Sensitivity (Recall)
					SP = 0.		# Specificity
					PC = 0. 	# Precision
					F1 = 0.		# F1 Score
					JS = 0.		# Jaccard Similarity
					DC = 0.		# Dice Coefficient
					length=0
					for i, (images, GT) in enumerate(self.valid_loader):
						images = images.to(self.device)
						GT = GT.to(self.device)

						SR = self.unet(images)
						acc += get_accuracy(SR,GT)

						SE += get_sensitivity(SR,GT)
						SP += get_specificity(SR,GT)
						PC += get_precision(SR,GT)
						F1 += get_F1(SR,GT)
						JS += get_JS(SR,GT)
						DC += get_DC(SR,GT)
						
						length += images.size(0)
					
					acc = acc/length
					SE = SE/length
					SP = SP/length
					PC = PC/length
					F1 = F1/length
					JS = JS/length
					DC = DC/length
					
					print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(acc,SE,SP,PC,F1,JS,DC))


					torchvision.utils.save_image(images.data.cpu(),
												os.path.join(self.result_path,
															'%s_valid_%d_image.png'%(self.model_type,epoch+1)))
					torchvision.utils.save_image(SR.data.cpu(),
												os.path.join(self.result_path,
															'%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
					torchvision.utils.save_image(GT.data.cpu(),
												os.path.join(self.result_path,
															'%s_valid_%d_GT.png'%(self.model_type,epoch+1)))

				'''	
				# Save Best style encoder model
				if (style_acc.data.cpu().numpy()/(i+1)) > best_se_acc:
					best_se_acc = style_acc.data.cpu().numpy()/(i+1)
					# Save the model parameters
					best_enc_style = self.enc_style.state_dict()
					print('Style encoder valid accuracy : %.2f'%best_se_acc)
				if (char_acc.data.cpu().numpy()/(i+1)) > best_ce_acc:
					best_ce_acc = char_acc.data.cpu().numpy()/(i+1)
					# Save the model parameters
					best_enc_char = self.enc_char.state_dict()
					print('Char encoder valid accuracy : %.2f'%best_ce_acc)
				'''
			# torch.save(best_enc_style, se_path)
			# print('Save style encoder at accuracy %.2f'%best_se_acc)
			# del best_enc_style, best_se_acc
			# torch.save(best_enc_char, ce_path)
			# print('Save char encoder at accuracy %.2f'%best_ce_acc)
			# del best_enc_char, best_ce_acc
			# del self.valid_loader, self.decoder

		"""
		#======================================= Main-Train ======================================#
		#=========================================================================================#
		start_iter = self.restore_model(self.start_epochs)
		self.num_epochs += start_iter

		# Main G/D Train
		iters_per_epoch = len(self.train_loader)
		g_lr = self.g_lr
		d_lr = self.d_lr
		start_time = time.time()
		for epoch in range(start_iter, self.num_epochs):
			self.generator.train(True)
			self.discriminator.train(True)
			for i, (x_style, x_target, style, char, x_style_c, x_char_s) in enumerate(self.train_loader):

				loss = {}
				batch_size = x_trg.size(0)
				# Generate real labels
				x_trg = self.to_variable(x_trg)
				style = self.to_variable(style)
				char = self.to_variable(char)
				char_onehot = self.label2onehot(char)
				# Character transfer. keep style. and thats' character index
				x_style = self.to_variable(x_style)
				x_style_c = self.to_variable(x_style_c)
				x_style_c_onehot = self.label2onehot(x_style_c)

				#==================================== Train D ====================================#

 				# Generate fake image from Encoder
				real_style, _ = self.enc_style(x_style)
				real_char, _ = self.enc_char(x_style)
				fake_img = self.generator(x_style_c_onehot, real_style, real_char, char_onehot)

				# 1) Train D to recognize real images as real.
				out_src, out_style, out_char = self.discriminator(x_trg)
				d_loss_real = torch.mean((out_src - 1) ** 2)		# Least square GANs

				# 2) Traing D to classify G(E(i)) as correct style/char
				d_loss_style = self.celoss(out_style, style)
				d_loss_char = self.celoss(out_char, char)

				# 3) Train D to recognize fake images as fake.
				fake_src, _,_ = self.discriminator(fake_img)
				d_loss_fake  = torch.mean(fake_src ** 2)			# Least Square GANs

				# 4) Compute loss for gradient penalty.
				alpha = torch.rand(x_style.size(0), 1, 1, 1).to(self.device)
				x_hat = (alpha * x_style.data + (1 - alpha) * fake_img.data).requires_grad_(True)
				out_src,_,_ = self.discriminator(x_hat)
				d_loss_gp = self.gradient_penalty(out_src, x_hat)

				# Compute gradient penalty
				d_loss = d_loss_real + d_loss_fake + \
						 self.lambda_cls * d_loss_style + self.lambda_cls * d_loss_char + \
						 self.lambda_gp * d_loss_gp

				# Logging
				loss['D/loss_real'] = d_loss_real.item()
				loss['D/loss_fake'] = d_loss_fake.item()
				loss['D/loss_style'] = d_loss_style.item()
				loss['D/loss_char'] = d_loss_char.item()
				loss['D/loss_gp'] = d_loss_gp.item()

				# Backward + Optimize
				self.reset_grad()
				d_loss.backward()
				self.d_optimizer.step()

				#==================================== Train G ====================================#
				if (i+1) % self.d_train_repeat == 0:

					# Generate fake image from Encoder
					real_style, _ = self.enc_style(x_style)
					real_char, _ = self.enc_char(x_style)

					fake_img = self.generator(x_style_c_onehot, real_style, real_char, char_onehot)

					# Generate identity image from Encoder
					id_img = self.generator(x_style_c_onehot, real_style, real_char, x_style_c_onehot)

					# Reconstruct Image from fake image
					fake_style, _ = self.enc_style(fake_img)
					rec_img = self.generator(char_onehot, fake_style, real_char, x_style_c_onehot)

					rec_style, _ = self.enc_style(rec_img)
					rec_char, _ = self.enc_char(rec_img)

					# 1) Train G so that D recognizes G(E(i)) as real.
					out_src, out_style, out_char = self.discriminator(fake_img)
					g_loss_fake = torch.mean((out_src - 1) ** 2)

					# 2) Training G so that D classifies G(E(i)) as correct style/char
					g_loss_style = self.celoss(out_style, style)
					g_loss_char = self.celoss(out_char, char)

					# 3) Training G to G(E(G(E(i)))) and (i) are similar. Reconstruct Loss
					g_loss_rec = torch.mean(torch.abs(x_style - rec_img))

					# 3.5.2) Training G to E(G(E(G(E(i))))) and E(i) are similar. Recons_Perceptual Loss
					g_loss_per_style = torch.mean((real_style - rec_style) ** 2)
					g_loss_per_char = torch.mean((real_char - rec_char) ** 2)
					g_loss_rec_per = g_loss_per_style + g_loss_per_char

					# 4) Training G to G(E(i)) and (i) are similar. Identity loss
					g_loss_id = torch.mean(torch.abs(x_style - id_img))

					# 6) Training G to 'x_style' and G(E(i)) are similar. L1 loss
					g_loss_l1 = torch.mean(torch.abs(x_trg - fake_img))

					# Compute Structural similarity measure of the Generator
					ssim_fake = utils.ssim(x_trg, fake_img)

					# Compute gradient penalty
					g_loss = g_loss_fake + \
							 self.lambda_cls * (g_loss_style + g_loss_char) + \
					 	     self.lambda_rec * (g_loss_rec + g_loss_rec_per
							 + g_loss_l1 - ssim_fake) + g_loss_id
					 	     #self.lambda_rec * (g_loss_rec + g_loss_rec_per) + \

					# Backprop + Optimize
					self.reset_grad()
					g_loss.backward()
					self.g_optimizer.step()

					# Logging
					loss['G/loss_fake'] = g_loss_fake.item()
					loss['G/loss_style'] = g_loss_style.item()
					loss['G/loss_char'] = g_loss_char.item()
					loss['G/loss_rec'] = g_loss_rec.item()
					loss['G/loss_id'] = g_loss_id.item()
					loss['G/loss_l1'] = g_loss_l1.item()
					loss['G/loss_ssim'] = 1.-ssim_fake.item()

					# Print the log info
					if (i+1) % self.log_step == 0:
						elapsed = time.time() - start_time
						elapsed = str(datetime.timedelta(seconds=elapsed))

						log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
								elapsed, epoch+1, self.num_epochs, i+1, iters_per_epoch)

						for tag, value in loss.items():
							log += ", {}: {:.4f}".format(tag, value)
						print(log)

						# Save real images
						x_trg = x_trg.view(x_trg.size(0), 2, self.image_size, self.image_size)
						x_trg = self.tensor2img(x_trg)
						fake_img = self.tensor2img(fake_img)
						torchvision.utils.save_image(self.denorm(np.reshape(x_trg.data.cpu(),(-1,1,self.image_size,self.image_size))),
							os.path.join(self.sample_path,
										'real_images-%d_train.png' %(epoch+1)))

						# save the sampled images
						torchvision.utils.save_image(self.denorm(np.reshape(fake_img.data.cpu(),(-1,1,self.image_size,self.image_size))),
							os.path.join(self.sample_path,
										'fake_samples-%d_train.png' %(epoch+1)))

			del x_trg, x_style, style, char, x_style_c, x_char_s
			del fake_img, rec_img, id_img, ssim_fake
			#==================================== Valid GD ====================================#
			if (epoch+1) % self.val_step == 0:
				self.generator.train(False)
				self.discriminator.train(False)
				self.generator.eval()
				self.discriminator.eval()

				ssim_fake = 0.
				ssim_rec = 0.
				for i, (x_style, _, x_trg, _, x_style_c, char, _) in enumerate(self.test_loader):

					batch_size = x_style.size(0)

					x_style = self.to_variable(x_style)
					x_trg = self.to_variable(x_trg)
					x_style_c = self.label2onehot(x_style_c)
					_, char = self.enc_char(x_trg)
					_, char = torch.max(char, dim=1)
					char = self.label2onehot(char)

					# Generate fake image from Encoder
					real_style, _  = self.enc_style(x_style)
					real_char, _ = self.enc_char(x_style)
					fake_img = self.generator(x_style_c, real_style, real_char, char)

					ssim_fake += utils.ssim(x_trg, fake_img).data.cpu().numpy()

					fake_img = self.generator(x_style_c, real_style, real_char, x_style_c)

					ssim_rec += utils.ssim(x_style, fake_img).data.cpu().numpy()

				print('Valid SSIM: ', end='')
				print("{:.4f}".format(ssim_fake/(i+1)))
				print('Reocn SSIM: ', end='')
				print("{:.4f}".format(ssim_rec/(i+1)))

				x_trg = x_trg.view(x_trg.size(0), 2, self.image_size, self.image_size)
				x_trg = self.tensor2img(x_trg)
				fake_img = self.tensor2img(fake_img)

				# Save real images
				torchvision.utils.save_image(np.reshape(x_trg.data.cpu(),(-1,1,self.image_size,self.image_size)),
					os.path.join(self.sample_path,
								'real_images-%d_test.png' %(epoch+1)))
				# save the sampled images
				torchvision.utils.save_image(np.reshape(fake_img.data.cpu(),(-1,1,self.image_size,self.image_size)),
					os.path.join(self.sample_path,
								 'fake_samples-%d_test.png' %(epoch+1)))


				# Save the model parameters for each epoch
				g_path = os.path.join(self.model_path, 'generator-%d.pkl' %(epoch+1))
				d_path = os.path.join(self.model_path, 'discriminator-%d.pkl' %(epoch+1))
				se_path = os.path.join(self.model_path, 'style_encoder-%d.pkl' %(epoch+1))
				ce_path = os.path.join(self.model_path, 'char_encoder-%d.pkl' %(epoch+1))
				torch.save(self.generator.state_dict(), g_path)
				torch.save(self.discriminator.state_dict(), d_path)
				torch.save(self.enc_style.state_dict(), se_path)
				torch.save(self.enc_char.state_dict(), ce_path)

			# Decay learning rate
			if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
				g_lr -= (self.g_lr / float(self.num_epochs_decay))
				d_lr -= (self.d_lr / float(self.num_epochs_decay))
				self.update_lr(g_lr, d_lr)
				print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
		"""
	
	'''
	#================================ Sampling ====================================/
	#==============================================================================/
	def sample(self):

		"""Translate images using StarGAN trained on a single dataset."""
		se_path = os.path.join(self.model_path, 'style_encoder-%d.pkl' %(self.sample_epochs))
		ce_path = os.path.join(self.model_path, 'char_encoder-%d.pkl' %(self.sample_epochs))

		# Style Encoder Train First
		if os.path.isfile(se_path) and os.path.isfile(ce_path):
			# Load the pretrained Encoder
			self.enc_style.load_state_dict(torch.load(se_path))
			self.enc_char.load_state_dict(torch.load(ce_path))
			print('Enocder is Successfully Loaded from %s'%se_path)

		# Load the trained generator.
		self.restore_model(self.sample_epochs)

		# Set data loader.
		data_loader = self.test_loader

		ssim_fake_total = 0.
		ssim_reco_total = 0.
		with torch.no_grad():
			for i, (x_style, x_trg,  _, x_style_c, char_trg, _) in enumerate(self.test_loader):

				batch_size = x_style.size(0)

				# Prepare input images and target domain labels.
				x_style = x_style.to(self.device)
				x_trg = x_trg.to(self.device)
				x_style_c = self.label2onehot(x_style_c)
				char_trg = self.label2onehot(char_trg)

				real_style, _ = self.enc_style(x_style)
				real_char, _ = self.enc_char(x_style)

				# Translate images.
				x_fake_list = [x_style, x_trg]
				x_fake = self.generator(x_style_c, real_style, real_char, char_trg)
				x_fake_list.append(x_fake)

				ssim_fake = utils.ssim(x_trg, x_fake).data.cpu().numpy()
				ssim_fake_total += ssim_fake

				print('Valid SSIM: ', end='')
				print("{:.4f}".format(ssim_fake))

				# Translate images.
				x_fake = self.generator(x_style_c, real_style, real_char, x_style_c)
				x_fake_list.append(x_fake)

				ssim_fake = utils.ssim(x_style, x_fake).data.cpu().numpy()
				ssim_reco_total += ssim_fake

				print('Recon SSIM: ', end='')
				print("{:.4f}".format(ssim_fake))
				# Save the translated images.
				x_concat = torch.cat(x_fake_list, dim=2)
				x_concat = self.tensor2img(x_concat)
				result_path = os.path.join(self.result_path, '{}-images.jpg'.format(i+1))
				torchvision.utils.save_image(np.reshape(x_concat.data.cpu(), (-1,1,self.image_size*4,self.image_size)),
					result_path, nrow=batch_size, padding=0)
			print('Saved real and fake images into {}...'.format(result_path))
			print('Total Valid SSIM: ', end='')
			print("{:.4f}".format(ssim_fake_total/(i+1)))
			print('Total Recon SSIM: ', end='')
			print("{:.4f}".format(ssim_reco_total/(i+1)))
	'''