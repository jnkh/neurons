from __future__ import print_function
import convnet as c
import numpy as np
import matplotlib.pyplot as plt
from build_model import build_model




def visualize_outputs(x,y,y_pred,idx=-1):
	fig = plt.figure()
	fig.add_subplot(1,3,1)
	eps = 1e-3
	# x = np.log10(x + eps)
	# y = np.log10(y + eps)
	# y_pred = np.log10(y_pred + eps)
	imgplot = plt.imshow(x,vmin=0,vmax=1)
	plt.title('raw')
	fig.add_subplot(1,3,2)
	imgplot = plt.imshow(y,vmin=0,vmax=1)
	plt.title('label')
	fig.add_subplot(1,3,3)
	imgplot = plt.imshow(y_pred,vmin=0,vmax=1)
	plt.title('predict')
	# plt.colorbar()
	if idx > 0:
		plt.savefig('patch_{}.png'.format(idx),bbox_inches='tight')
	plt.show()



def visualize_result(model,x,y,image_size,thresh,idx=-1):
	y_pred = model.predict(x,batch_size=1)
	y_pred = np.reshape(y_pred,image_size)
	mask = y_pred > thresh
	y_pred[mask] = 1.0
	y_pred[~mask] = 0.0
	y = np.reshape(y,image_size)
	x = np.reshape(x,image_size)
	visualize_outputs(x,y,y_pred,idx)



def compute_pixel_loss(y,y_pred,thresh):
	y = y > 0.5
	y_pred = y_pred > thresh

	TP = 1.0*np.sum(np.logical_and(y,y_pred))
	TN = 1.0*np.sum(np.logical_and(~y,~y_pred))
	FP = 1.0*np.sum(np.logical_and(~y,y_pred))
	FN = 1.0*np.sum(np.logical_and(y,~y_pred))

	if TP + FN == 0:
		tpr = 0.0
	else:
		tpr = TP/(TP + FN)

	fpr = FP/(FP + TN) 
	ave = np.mean(y)
	if thresh == 0.0:
		print(ave)
	return tpr,fpr,ave

def roc_curve(y,y_pred):
	thresh_range = np.logspace(-4,0,200)
	fp_range = np.zeros_like(thresh_range)
	tp_range = np.zeros_like(thresh_range)
	for i,thresh in enumerate(thresh_range):
		tpr,fpr,_ = compute_pixel_loss(y,y_pred,thresh)
		fp_range[i] = fpr
		tp_range[i] = tpr
	plt.plot(1-fp_range,tp_range)
	plt.xlabel('1-FP')
	plt.ylabel('TP')
	plt.xlim([-0.01,1.01])
	plt.ylim([-0.01,1.01])
	plt.grid()
	plt.figure()
	plt.semilogx(thresh_range,fp_range,'r')
	plt.semilogx(thresh_range,tp_range,'g')
	plt.show()



def overall_roc_curve(model,X,y):
	perm = np.random.permutation(np.arange(len(X)))
	num = len(X)
	X = X[perm][:num]
	y = y[perm][:num]

	y_pred = model.predict(X,verbose=1)
	roc_curve(y,y_pred)


image_size = (228,228)
full_image_size = (1,image_size[0],image_size[1])

#training
#path = '../data/image_zip_pairs/l7cre_ts01_20150928_005na_z3um_3hfds_561_70msec_5ovlp_C00_Z0978/'
#testing
path = '../data/image_zip_pairs/20160211_vc22_01_lob5_70msec_z3um_3hfds_561_C00_Z1558/'
X,y = c.parser(path,image_size)
X,y = c.normalize(X,y,full_image_size)

print(np.median(X),np.mean(y))


save_path = 'weights.h5'

model = build_model(full_image_size)
model.load_weights(save_path)

shown = 0
thresh = 0.00129

#overall_roc_curve(model,X,y)
while shown < 5:
	sample = np.random.randint(0,X.shape[0])
	X_curr = X[sample:sample+1]
	y_curr = y[sample:sample+1]
	print('sample: {}, max: {}'.format(sample,np.max(y_curr)))
	if np.max(y_curr) > 0.5:
		visualize_result(model,X_curr,y_curr,image_size,thresh,idx=sample)
		shown += 1





