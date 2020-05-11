'''
Created on Mon Sep 06 20:55:06 2018

Author: Fatemeh Zabihollahy
'''
#%%
import numpy
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import scipy
from skimage import morphology
from keras.models import Model, load_model
from keras.layers.merge import concatenate
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers.normalization import BatchNormalization
import nibabel as nib
import glob
from matplotlib import pyplot as plt

path1 = r'C:\Users\Fatemeh\Desktop\LGE Cardiac MRI\LGE Images nii'
LGEs = glob.glob(path1 + "/*")

path2 = r'C:\Users\Fatemeh\Desktop\LGE Cardiac MRI\Myocardial Masks nii'
MYOs = glob.glob(path2 + "/*")


#%%

x_unet = 256
y_unet = 256

data_train = numpy.zeros((1,x_unet*y_unet))
mask_train =  numpy.zeros((1,x_unet*y_unet))

for n in range(14): 
    
    data_lge = nib.load(LGEs[n]);
    lge = data_lge.get_data()
    x,y,z = lge.shape
    
    lge_norm = numpy.zeros((x,y,z))
    for slice_no in range (z):
        lge_slice = lge[:, :, slice_no]
        for a in range (x):
            for b in range (y):
                if lge_slice[a,b] > 1000:
                    lge_slice[a,b] = numpy.median(lge_slice)
        lge_slice = (lge_slice-lge_slice.min())/(lge_slice.max()-lge_slice.min())
        lge_norm[:, :, slice_no] = lge_slice
       
    
    data_myo = nib.load(MYOs[n]);
    myo = data_myo.get_data()
    
#%
    data = numpy.zeros((1,x_unet*y_unet))
    mask = numpy.zeros((1,x_unet*y_unet))
    
    x_pad = int(x_unet - x)
    y_pad = int(y_unet - y)
    
    for page in range(0,z):    
        lge_slice = lge_norm[:,:,page]
        myo_slice = myo[:,:,page]
        
        if (numpy.max(myo_slice) != 0):
            lge_slice = numpy.pad(lge_slice, ((0, x_pad),(0, y_pad)), 'wrap')
            myo_slice = numpy.pad(myo_slice, ((0, x_pad),(0, y_pad)), 'wrap')
            
          
            
            lge_flipud = numpy.flipud(lge_slice)
            myo_flipud = numpy.flipud(myo_slice)
           
            
            lge_slice = lge_slice.reshape(1,(x_unet*y_unet))
            myo_slice = myo_slice.reshape(1, (x_unet*y_unet)) 
            
            lge_fud_reshape = lge_flipud.reshape(1,(x_unet*y_unet))
            myo_fud_reshape = myo_flipud.reshape(1,(x_unet*y_unet))
            
            data = numpy.vstack((data, lge_slice, lge_fud_reshape))
            mask = numpy.vstack((mask,myo_slice, myo_fud_reshape))
          

    data = numpy.delete(data, (0), axis=0)     
    data_train = numpy.vstack((data_train, data))   
    
    mask = numpy.delete(mask, (0), axis=0)     
    mask_train = numpy.vstack((mask_train, mask)) 
        
data_train = numpy.delete(data_train, (0), axis=0) 
mask_train = numpy.delete(mask_train, (0), axis=0) 

  
#% reshape training dataset
data_train = data_train.reshape(data_train.shape[0], x_unet, y_unet, 1)
mask_train = mask_train.reshape(mask_train.shape[0], x_unet, y_unet, 1)
#%% Visualize one sample of the training data and mask
k = 260
lge_sample = data_train[k,:]
lge_sample = lge_sample.reshape(x_unet, y_unet)
lge_img = Image.fromarray(lge_sample*255)
lge_img.show()

mask_sample = mask_train[k,:]
mask_sample = mask_sample.reshape(x_unet, y_unet)
mask_img = Image.fromarray(mask_sample*255)
mask_img.show()
#%% U-net1 Architecture
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


filter_no = 32

inputs = Input((x_unet, y_unet, 1))


conv1 = Conv2D(filter_no, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(inputs)
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(filter_no, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv1)
conv1 = BatchNormalization()(conv1)


pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
conv2 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(pool1)
conv2 = BatchNormalization()(conv2)
conv2 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv2)
conv2 = BatchNormalization()(conv2)


pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
conv3 = Conv2D(filter_no*4, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(pool2)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(filter_no*4, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv3)
conv3 = BatchNormalization()(conv3)


pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
conv4 = Conv2D(filter_no*8, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(pool3)
conv4 = BatchNormalization()(conv4)
conv4 = Conv2D(filter_no*8, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv4)
conv4 = BatchNormalization()(conv4)
conv4 = Dropout(0.5)(conv4)

pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)
conv5 = Conv2D(filter_no*16, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(pool4)
conv5 = BatchNormalization()(conv5)
conv5 = Conv2D(filter_no*16, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv5)
conv5 = BatchNormalization()(conv5)


up1 = UpSampling2D(size = (2,2))(conv5)
merge1 = concatenate([conv4,up1], axis = 3)
conv6 = Conv2D(filter_no*8, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(merge1)
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(filter_no*8, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv6)
conv6 = BatchNormalization()(conv6)


up2 = UpSampling2D(size = (2,2))(conv6)
merge2 = concatenate([conv3,up2], axis = 3)
conv7 = Conv2D(filter_no*4, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(merge2)
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(filter_no*4, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv7)
conv7 = BatchNormalization()(conv7)


up3 = UpSampling2D(size = (2,2))(conv7)
merge3 = concatenate([conv2,up3], axis = 3)
conv8 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(merge3)
conv8 = BatchNormalization()(conv8)
conv8 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv8)
conv8 = BatchNormalization()(conv8)


up4 = UpSampling2D(size = (2,2))(conv8)
merge4 = concatenate([conv1,up4], axis = 3)
conv9 = Conv2D(filter_no, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(merge4)
conv9 = BatchNormalization()(conv9)
conv9 = Conv2D(filter_no, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv9)
conv9 = BatchNormalization()(conv9)
conv9 = Conv2D(2, 3, strides=(1, 1), activation = 'relu', padding = 'same')(conv9)
conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

model = Model(input = inputs, output = conv9)

model.compile(optimizer='adadelta', loss=dice_coef_loss, metrics=[dice_coef])
model.summary()

#%% U-net2 Architecture
'''
filter_no = 16
inputs = Input((x_unet, y_unet, 1))

conv1 = Conv2D(filter_no, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(inputs)
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(filter_no, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv1)
conv1 = BatchNormalization()(conv1)

pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
conv2 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(pool1)
conv2 = BatchNormalization()(conv2)
conv2 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv2)
conv2 = BatchNormalization()(conv2)

pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
conv3 = Conv2D(filter_no*4, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(pool2)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(filter_no*4, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv3)
conv3 = ZeroPadding2D(padding=(1, 1))(conv3)
conv3 = BatchNormalization()(conv3)
conv3 = Dropout(0.5)(conv3)

pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
conv4 = Conv2D(filter_no*8, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(pool3)
conv4 = BatchNormalization()(conv4)
conv4 = Conv2D(filter_no*8, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv4)
conv4 = BatchNormalization()(conv4)
conv4 = Dropout(0.5)(conv4)

up1 = UpSampling2D(size = (2,2))(conv4)
merge1 = concatenate([conv3,up1], axis = 3)
conv5 = Conv2D(filter_no*4, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(merge1)
conv5 = BatchNormalization()(conv5)
conv5 = Conv2D(filter_no*4, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv5)
conv5 = BatchNormalization()(conv5)

up2 = UpSampling2D(size = (2,2))(conv5)
up2 = Cropping2D(cropping=((2, 2), (2, 2)))(up2)
merge2 = concatenate([conv2,up2], axis = 3)
conv6 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(merge2)
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv6)
conv6 = BatchNormalization()(conv6)

up3 = UpSampling2D(size = (2,2))(conv6)
merge3 = concatenate([conv1,up3], axis = 3)
conv7 = Conv2D(filter_no, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(merge3)
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(filter_no, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv7)
conv7= BatchNormalization()(conv7)
conv7 = Conv2D(2, 3, strides=(1, 1), activation = 'relu', padding = 'same')(conv7)
conv7 = Conv2D(1, 1, activation = 'sigmoid')(conv7)

model = Model(input = inputs, output = conv7)

#sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#model.compile(loss='mean_squared_error', optimizer=sgd, metrics = ['accuracy'])

#model.compile(optimizer = 'adadelta', loss = 'binary_crossentropy', metrics = ['accuracy'])	
model.compile(optimizer='adadelta', loss=dice_coef_loss, metrics=[dice_coef])
model.summary()
'''
#%% Train Model
#fname= "scar_fulauto_unet1.hdf5"
#earlystopper = EarlyStopping(patience=20, verbose=1)
#checkpointer = ModelCheckpoint('myo_fcn1.hdf5', verbose=1, save_best_only=True)
results = model.fit(data_train, mask_train, validation_split=0.2, shuffle=True, batch_size=10, epochs=100)	
#%%
# summarize history for accuracy
plt.plot(results.history['dice_coef'])
plt.plot(results.history['val_dice_coef'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#%% Save trained model and the network weights
#fname= "myo_fcn3.hdf5"
model.save(fname, overwrite = True)
#model.save_weights("MYO_seg_unet3_weights.h5")

#%%
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
fname= "myo_fcn1.hdf5"
model = load_model(fname, custom_objects={'dice_coef': dice_coef,'dice_coef_loss': dice_coef_loss})
#%%
'''
def Plot_Results1(img_test,seg_clean,mask_clean):
    
    img_test = img_test.reshape( x_unet, y_unet)
    img_test = img_test[:x, :y]

    fig, axes = plt.subplots(1, 3, figsize=(8, 8))
    ax = axes.flatten()
    
    ax[0].imshow(img_test.reshape(x,y), cmap="gray")
    ax[0].set_axis_off()
    ax[0].contour(seg_clean, [0.5], colors='r')
    ax[0].contour(mask_clean, [0.5], colors='b')
    #ax[0].set_title("Morphological ACWE segmentation", fontsize=12)
    
    ax[1].imshow(seg_clean, cmap="gray")
    ax[2].imshow(mask_clean, cmap="gray")
'''    
    
def Plot_Results(img_test,seg_clean,mask_clean):
   
    color_myo = numpy.dstack((seg_clean,numpy.zeros((x,y)),numpy.zeros((x,y)))) 
    
    #img_test = img_test.reshape(x_unet,y_unet)
     
    plt.figure()
    plt.imshow(img_test, cmap="gray")
    plt.imshow(color_myo, 'gray', interpolation='none', alpha=0.4)
    #ax[0].imshow(color_tz, 'gray', interpolation='none', alpha=0.3)
    plt.contour(mask_clean, [2], colors='c')
    

def model_evaluate(data,mask):
    dsc = []
    acc = []
    prec = []
    rec = []
    vol_manual = []
    vol_seg = []

    for k in range(len(data)):
        mask_sample = mask[k,:,:,:]
        mask_sample = mask_sample.reshape(x_unet, y_unet)
        img_test = data[k,:, :, :]
        img_test = img_test.reshape(1, x_unet, y_unet, 1)
        img_pred = model.predict(img_test, batch_size=1, verbose=1)
        img_pred = img_pred.reshape(x_unet, y_unet)
        img_pred  = (img_pred  > 0.5).astype(numpy.uint8)
        
        seg_clean = numpy.array(img_pred, bool)
        seg_clean = morphology.remove_small_objects(seg_clean,100) 
        seg_clean = seg_clean*1 
        
        seg_clean = scipy.ndimage.morphology.binary_dilation(seg_clean, iterations=3)
        seg_clean = scipy.ndimage.morphology.binary_erosion(seg_clean)
        #seg_clean = scipy.ndimage.morphology.binary_fill_holes(seg_clean) 
        seg_clean = seg_clean*1    
        seg_clean = seg_clean[:x, :y]
        
        mask_clean = scipy.ndimage.morphology.binary_dilation(mask_sample,  iterations=2)
        mask_clean = scipy.ndimage.morphology.binary_erosion(mask_clean)
        #mask_clean = scipy.ndimage.morphology.binary_fill_holes(mask_clean) 
        mask_clean = mask_clean*1
        mask_clean = mask_clean[:x, :y]
        
        img_test = img_test.reshape( x_unet, y_unet)
        img_test = img_test[:x, :y]
        lge_final[:,:,k] = img_test
        myo_final[:,:,k] = seg_clean
        gt_final[:,:,k] = mask_clean
        
        #seg_clean = morphological_chan_vese(img_pred, 5, init_level_set=seg_clean, smoothing=2)
        Plot_Results(img_test,seg_clean,mask_clean)
                
        y_true = numpy.reshape(mask_clean, (x*y,1))
        y_pred = numpy.reshape(seg_clean, (x*y,1))
        
        dsc = numpy.append(dsc,f1_score(y_true, y_pred, average='macro'))
        acc = numpy.append(acc,accuracy_score(y_true, y_pred))
        prec = numpy.append(prec,precision_score(y_true, y_pred, average='macro'))
        rec = numpy.append(rec,recall_score(y_true, y_pred, average='macro'))
        vol_manual =  numpy.append(vol_manual,numpy.sum(y_true)*1.3*0.625*0.625/1000) 
        vol_seg =  numpy.append(vol_seg,numpy.sum(y_pred)*1.3*0.625*0.625/1000) 
    
    dsc = round(numpy.median(dsc)*100,2)
    acc = round(numpy.median(acc)*100,2)
    prec = round(numpy.median(prec)*100,2)
    rec = round(numpy.median(rec)*100,2)
    vol_manual = numpy.sum(vol_manual)
    vol_seg = numpy.sum(vol_seg)
    return(dsc,acc,prec,rec,vol_manual,vol_seg)
#%% Create test dataset including test images and their corresponding masks
import time
x_unet = 256
y_unet = 256

dice_index =[]
accuracy = []
precision = []
recall = []
sec = []
v_m = []
v_a = []

slice_total = 0

for n in range(21,22): 
    start_time = time.time()
    data_lge = nib.load(LGEs[n]);
    lge = data_lge.get_data()
    x,y,z = lge.shape
    
    lge_final = numpy.zeros((x,y,z))
    myo_final = numpy.zeros((x,y,z))
    gt_final = numpy.zeros((x,y,z))
    
    lge_norm = numpy.zeros((x,y,z))
    for slice_no in range (z):
        lge_slice = lge[:, :, slice_no]
        for a in range (x):
            for b in range (y):
                if lge_slice[a,b] > 1000:
                    lge_slice[a,b] = numpy.median(lge_slice)
        lge_slice = (lge_slice-lge_slice.min())/(lge_slice.max()-lge_slice.min())
        lge_norm[:, :, slice_no] = lge_slice
      
    #lge = block_reduce(lge, block_size=(downsample_factor, downsample_factor,1), func=numpy.mean)  
    
    data_myo = nib.load(MYOs[n]);
    myo = data_myo.get_data()    
#% 
    data = numpy.zeros((1,x_unet*y_unet))
    mask = numpy.zeros((1,x_unet*y_unet))
    
    x_pad = int(x_unet - x)
    y_pad = int(y_unet - y)
    
    for page in range(0,z):    
        lge_slice = lge_norm[:,:,page]
        myo_slice = myo[:,:,page]
        
        if (numpy.max(myo_slice) != 0):
            lge_slice = numpy.pad(lge_slice, ((0, x_pad),(0, y_pad)), 'wrap')
            myo_slice = numpy.pad(myo_slice, ((0, x_pad),(0, y_pad)), 'wrap')
            
            lge_slice = lge_slice.reshape(1,(x_unet*y_unet))
            myo_slice = myo_slice.reshape(1, (x_unet*y_unet)) 
            
            data = numpy.vstack((data,lge_slice ))
            mask = numpy.vstack((mask,myo_slice))

    data = numpy.delete(data, (0), axis=0)     
    
    mask = numpy.delete(mask, (0), axis=0)     
    
    data = data.reshape(data.shape[0], x_unet, y_unet, 1)
    mask = mask.reshape(mask.shape[0], x_unet, y_unet, 1)
        
    p1,p2,p3,p4,p5,p6 = model_evaluate(data,mask)
    dice_index = numpy.append(dice_index,p1)
    accuracy = numpy.append(accuracy,p2)
    precision = numpy.append(precision,p3)
    recall = numpy.append(recall,p4)
    slice_total += data.shape[0]
    v_m = numpy.append(v_m,p5)
    v_a = numpy.append(v_a,p6)
    sec =  numpy.append(sec,(time.time() - start_time))   
#data_sample_visualization(data, 15, mask)

#%      
print('Mean Values:')    
print('DI is :', round(numpy.mean(dice_index),2) , '+', round(numpy.std(dice_index),2))
print('Acc. is :', round(numpy.mean(accuracy),2), '+', round(numpy.std(accuracy),2))
print('Precision is :', round(numpy.mean(precision),2), '+', round(numpy.std(precision),2))
print('Recall is :', round(numpy.mean(recall),2), '+', round(numpy.std(recall),2))

print('Median Values:') 
print('DI is :', round(numpy.median(dice_index),2) , '+', round(numpy.std(dice_index),2))
print('Acc. is :', round(numpy.median(accuracy),2), '+', round(numpy.std(accuracy),2))
print('Precision is :', round(numpy.median(precision),2), '+', round(numpy.std(precision),2))
print('Recall is :', round(numpy.median(recall),2), '+', round(numpy.std(recall),2))


#%%  Applying a coloured mask overlay to an image 
'''
from skimage import data, color, io, img_as_float
import numpy as np
import matplotlib.pyplot as plt

k = 60
    
img = lge_norm[:,:,k]    
mask = myo_final[:,:,k]    
color_mask = np.dstack((mask,numpy.zeros((x,y)),numpy.zeros((x,y))))

#color_mask[30:140, 30:140] = [1, 0, 0]  # Red block
#color_mask[170:270, 40:120] = [0, 1, 0] # Green block
#color_mask[200:350, 200:350] = [0, 0, 1] # Blue block

img_color = np.dstack((img,img,img))

img_hsv = color.rgb2hsv(img_color)
color_mask_hsv = color.rgb2hsv(color_mask)

img_hsv[..., 0] = color_mask_hsv[..., 0]
img_hsv[..., 1] = color_mask_hsv[..., 1] * 0.7

img_masked = color.hsv2rgb(img_hsv)

# Display the output
f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 8),
                                  subplot_kw={'xticks': [], 'yticks': []})
ax0.imshow(img, cmap=plt.cm.gray)
ax1.imshow(color_mask)
ax2.imshow(img_masked)
plt.show() 
'''   
#%% Create a NifTi file from a numpy array

lge_orig = nib.Nifti1Image(lge_final, data_lge.affine, data_lge.header)
nib.save(lge_orig, "LGE-529.nii.gz")    

myo_gt = nib.Nifti1Image(gt_final, data_myo.affine, data_myo.header)
nib.save(myo_gt, "GT-myo529.nii.gz")    

myo_unet = nib.Nifti1Image(myo_final, data_myo.affine, data_myo.header)
nib.save(myo_unet, "U-net-myo529.nii.gz")    