import convnet as c
import numpy as np
import matplotlib.pyplot as plt
from build_model import build_model




image_size = (228,228)
full_image_size = (1,image_size[0],image_size[1])
path = '../data/image_zip_pairs/l7cre_ts01_20150928_005na_z3um_3hfds_561_70msec_5ovlp_C00_Z0978/'
X,y = c.parser(path,image_size)
X,y = c.normalize(X,y,full_image_size)


import keras
from keras.optimizers import Adam

save_path = 'weights.h5'



model = build_model(full_image_size)

opt = Adam(lr = 0.01)
model.compile(loss='binary_crossentropy', optimizer=opt)

model.fit(X,y,batch_size=16,nb_epoch=10,validation_split=0.1)

model.save_weights(save_path)

model.fit(X,y,batch_size=16,nb_epoch=10,validation_split=0.1)

model.save_weights(save_path)






