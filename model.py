
# coding: utf-8

# ## Create and train a DNN to predict steering angle for car in simulator

# ### Generator function to read the data from file

# #### Import necessary packages

# In[1]:


import csv
import numpy as np
import matplotlib.image as mpimg
import sklearn
from sklearn.model_selection import train_test_split


# In[2]:


# Directory containing the training data
dir_path = './data_jkinni/'


# #### Helper functions to read the training data

# In[9]:


def get_img_file_path(line, index, dir_path):
    """ Returns path to an image 
    using directory path, line from driving_log.csv and index.
    index: 0 for center image, 1 for left image and 2 for right image.  
    """
    orig_path = line[index]
    file_name = orig_path.split('\\')[-1]
    # orig_path.split('\\')[-2] is always /IMG/
    set_name = orig_path.split('\\')[-3]
    file_path = dir_path+'/'+set_name+'/'+'/IMG/'+file_name
    return file_path

def concat_dataset(base_path, set_name, lines):
    """ Concatenates a dataset to the lines of csv file.
    """
    csvpath = base_path + set_name + '/driving_log.csv'
    csvfile = open(csvpath)
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    return lines

def generator(dir_path, lines, batch_size=32):
    """ Image and steering angle generaor.
    """
    num_lines = len(lines)
    while 1:
        for offset in range(0, num_lines, batch_size):
            batch_lines = lines[offset:offset+batch_size]
            
            # Initialize image and angles
            images = []
            angles = []            
          
            # Adjustment applied to "guesstimate" the angle
            # for left and right images using center image angle
            side_angle_factor = 0.2
            
            # Get images and angles from lines
            for line in batch_lines:
                c_path = get_img_file_path(line, 0, dir_path)
                l_path = get_img_file_path(line, 1, dir_path)
                r_path = get_img_file_path(line, 2, dir_path)
                angle = float(line[3])
                
                # Read the images
                c_img = mpimg.imread(c_path)
                l_img = mpimg.imread(l_path)
                r_img = mpimg.imread(r_path)

                # Append the images and steers
                # For left and right images "guesstimate" the steers
                images.append(l_img)
                angles.append(angle + side_angle_factor)

                images.append(c_img)
                angles.append(angle)

                images.append(r_img)
                angles.append(angle - side_angle_factor)

            # Augment the data by flipping the image and negating the angle
            aug_images = []
            aug_angles = []
            for image, angle in zip(images, angles):
                aug_images.append(image)
                aug_angles.append(angle)
                image_flipped = np.fliplr(image)
                angle_flipped = -angle
                aug_images.append(image_flipped)
                aug_angles.append(angle_flipped)  

            X_train = np.array(aug_images)
            y_train = np.array(aug_angles)            
            yield sklearn.utils.shuffle(X_train, y_train)           


# #### Read the training data

# In[10]:


lines = []
# Read data of a forward lap around track1
lines = concat_dataset(dir_path, 'track1_forward_lap', lines)
# Read data of recovering from left side on track1
lines = concat_dataset(dir_path, 'track1_recovery_left', lines)
# Read data of recovering from right side on track1
lines = concat_dataset(dir_path, 'track1_recovery_right', lines)
# Read data of a reverse lap around track1
lines = concat_dataset(dir_path, 'track1_reverse_lap', lines)
# Read data of a reverse lap around track2
lines = concat_dataset(dir_path, 'track2_forward_lap', lines)


train_lines, val_lines = train_test_split(lines, test_size = 0.2)

train_generator = generator(dir_path, train_lines, batch_size=32)
validation_generator = generator(dir_path, val_lines, batch_size=32)


# #### Implement the network based on Nvidia DriveAV's model

# In[ ]:


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# Pre-Proessing (Normalization and Cropping)
model.add(Lambda(lambda x: (x/255.0 - 0.5), input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# 5 Convolution layers with Dropout(0.5) after first 2
model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))

# Flatten
model.add(Flatten())

# 5 Fully connected layers with Dropout(0.5) after each layer except output
model.add(Dense(200))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch=6*len(train_lines),
                    validation_data=validation_generator, nb_val_samples=6*len(val_lines),
                    nb_epoch=5, verbose=1)

model.save('model.h5')

