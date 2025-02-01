import numpy as np
import configparser

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, Convolution2D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from tensorflow.keras.optimizers import SGD

import sys
sys.path.insert(0, './lib/')
from Functions.help_functions import *

#function to obtain data for training/testing (validation)
from Functions.extract_patches import get_data_training

def Train():
    #Define the neural network
    def get_unet(n_ch,patch_height,patch_width):
        inputs = Input(shape=(n_ch,patch_height,patch_width))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
        conv1 = Dropout(0.2)(conv1)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)
        #
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
        conv2 = Dropout(0.2)(conv2)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)
        #
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
        conv3 = Dropout(0.2)(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)

        up1 = UpSampling2D(size=(2, 2))(conv3)
        up1 = concatenate([conv2,up1],axis=1)
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
        conv4 = Dropout(0.2)(conv4)
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
        #
        up2 = UpSampling2D(size=(2, 2))(conv4)
        up2 = concatenate([conv1,up2], axis=1)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
        conv5 = Dropout(0.2)(conv5)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
        #
        conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
        conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
        conv6 = core.Permute((2,1))(conv6)
        ############
        conv7 = core.Activation('softmax')(conv6)

        model = Model(inputs=inputs, outputs=conv7)

        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
        model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

        return model

    #========= Load settings from Config file
    config = configparser.RawConfigParser()
    config.read("Vessel-Tech-master/configuration.txt")
    print(config)
    #patch to the datasets
    path_data = config.get('data paths', 'path_local')
    #Experiment name
    name_experiment = config.get('experiment name', 'name')
    #training settings
    N_epochs = int(config.get('training settings', 'N_epochs'))
    batch_size = int(config.get('training settings', 'batch_size'))
    learning_rate = float(config.get('training settings', 'learning_rate'))
    momentum = float(config.get('training settings', 'momentum'))


    #============ Load the data and divided in patches
    patches_imgs_train, patches_masks_train = get_data_training(
        DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
        DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),  #masks
        patch_height = int(config.get('data attributes', 'patch_height')),
        patch_width = int(config.get('data attributes', 'patch_width')),
        N_subimgs = int(config.get('training settings', 'N_subimgs')),
        inside_FOV = config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
    )


    #========= Save a sample of what you're feeding to the neural network ==========
    N_sample = min(patches_imgs_train.shape[0],40)
    # Vessel-Tech-master\data\test
    visualize(group_images(patches_imgs_train[1000:1040,:,:,:],5),'Vessel-Tech-master/data/'+name_experiment+'/'+"sample_input_imgs")#.show()
    visualize(group_images(patches_masks_train[1000:1040,:,:,:],5),'Vessel-Tech-master/data/'+name_experiment+'/'+"sample_input_masks")#.show()


    #=========== Construct and save the model arcitecture =====
    n_ch = patches_imgs_train.shape[1]
    patch_height = patches_imgs_train.shape[2]
    patch_width = patches_imgs_train.shape[3]
    model = get_unet(n_ch, patch_height, patch_width)  #the U-net model
    print ("Check: final output of the network:")
    print (model.output_shape)
    #plot(model, to_file='Vessel-Tech-master/data/'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
    json_string = model.to_json()
    open('Vessel-Tech-master/data/'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)



    #============  Training ==================================
    checkpointer = ModelCheckpoint(filepath='Vessel-Tech-master/data/'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased
    print("finish checkpointer")

    patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption
    # It's just epochs now. nb_epoch was deprecated years ago.
    print("Finish U-net")
    print(len(patches_masks_train))
    model.fit(patches_imgs_train, patches_masks_train, epochs=N_epochs-145, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer])
    print("finish model fitting")


    #========== Save and test the last model ===================
    model.save_weights('Vessel-Tech-master/data/'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
    #test the model
    return 0
