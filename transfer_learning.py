#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 22:17:43 2019

@author: victor
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras
import tensorflow_datasets as tfds
import tempfile

tfds.disable_progress_bar()
# See available datasets
#print(tfds.list_builders())
list_ds={1:"cats_vs_dogs",2:"tf_flowers"}
list_model={1:"MobileNetV2",2:"resnet50"}
lits_opt={1:'sgd_momentum',2:'adam'}

data_sets=list_ds[2]
model_name=list_model[2]
opt_idx=lits_opt[2]

initial_epochs = 10
fine_tune_epochs = 10


tensorboard=tf.keras.callbacks.TensorBoard(log_dir='logs/model_{}_{}_{}'.format(model_name,data_sets,opt_idx), histogram_freq=1)


SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.ALL.subsplit(weighted=SPLIT_WEIGHTS)

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    data_sets,
    split=list(splits),
    data_dir='tensorflow_datasets/',
    with_info=True, as_supervised=True)


#assert isinstance(raw_train, tf.data.Dataset)
print('num_classes:', metadata.features['label'].num_classes) 
print('num_examples:', metadata.splits['train'].num_examples)

print(raw_train)
print(raw_validation)
print(raw_test)

get_label_name = metadata.features['label'].int2str

raw_train_length = [i for i,_ in enumerate(raw_train)][-1] + 1
raw_validation_length = [i for i,_ in enumerate(raw_validation)][-1] + 1
raw_test_length = [i for i,_ in enumerate(raw_test)][-1] + 1
total_data=raw_train_length+raw_validation_length+raw_test_length

print('raw_train_length:',raw_train_length)
print('raw_validation_length:',raw_validation_length)
print('raw_test_length:',raw_test_length)
print('total_data:',total_data)

for image, label in raw_train.take(2): #take(-1) to loop through all dataset
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))

labels=[]  
for image, label in raw_train.take(-1):
    labels.append(get_label_name(label))
  
IMG_SIZE = 160 if model_name=="MobileNetV2" else 100

def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1024



train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)


for image_batch, label_batch in train_batches.take(1):
   pass

image_batch.shape
label_batch.shape

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
if model_name=="MobileNetV2":
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
elif model_name=="resnet50":

    base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

feature_batch = base_model(image_batch)
print(feature_batch.shape)


base_model.trainable = False
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

NUM_CLASSES=1 if metadata.features['label'].num_classes==2 else metadata.features['label'].num_classes
prediction_layer = keras.layers.Dense(NUM_CLASSES, activation=None if metadata.features['label'].num_classes==2 else 'softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

#learning_rate= 0.01 
opt=tf.keras.optimizers.Adam(lr=1e-5) if opt_idx=='adam' else tf.keras.optimizers.SGD(lr=0.0005,momentum=0.9)
model.compile(optimizer=opt,
              loss='binary_crossentropy' if metadata.features['label'].num_classes==2 else 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
len(model.trainable_variables)


validation_steps = 20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)


print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches, callbacks=[tensorboard])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy',color='orange')
plt.plot(val_acc, label='Validation Accuracy',color='royalblue')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(0.5,min(plt.ylim())),1.1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss',color='orange')
plt.plot(val_loss, label='Validation Loss',color='royalblue')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('Figures/{}_{}_{}_change_classifier.png'.format(model_name,data_sets,opt_idx))
plt.show()

#%%=================Fine tunning=================================================
base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards

fine_tune_at = 100 if model_name=="MobileNetV2" else 143 


def add_regularization(model_reg, regularizer=tf.keras.regularizers.l2(0.001)):

    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
      print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
      return model_reg

    for layer in model_reg.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model_reg.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model_reg.save_weights(tmp_weights_path)

    # load the model from the config
    model_reg = tf.keras.models.model_from_json(model_json)
    
    # Reload the model weights
    model_reg.load_weights(tmp_weights_path, by_name=True)
    return model_reg


# freeze all layers up to stage 5 for resnet50,note BN is already frozen (ie running in test phase)
#reference https://github.com/PrzemekPobrotyn/CIFAR-10-transfer-learning/blob/master/report.ipynb
# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
  if 'bn' in layer.name:
    layer.freeze = False

#if model_name=="resnet50":   
#    model=add_regularization(model)
##     unfreeze BN layers from stage 5
#    for layer in model.layers[fine_tune_at:]:
#        if 'bn' in layer.name:
#            layer.freeze=False

model.compile(optimizer=opt,
              loss='binary_crossentropy' if metadata.features['label'].num_classes==2 else 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
len(model.trainable_variables)



total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_batches,
                         epochs=total_epochs,
                         initial_epoch =  history.epoch[-1],
                         validation_data=validation_batches, callbacks=[tensorboard])


acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

def write_(val,name):
    with open('results/'+name+'.txt', 'w') as output:
        output.write(str(val))

write_(acc,'{}_{}_{}_acc'.format(model_name,data_sets,opt_idx))
write_(val_acc,'{}_{}_{}_val_acc'.format(model_name,data_sets,opt_idx))
write_(loss,'{}_{}_{}_loss'.format(model_name,data_sets,opt_idx))
write_(val_loss,'{}_{}_{}_val_loss'.format(model_name,data_sets,opt_idx))


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy',color='orange')
plt.plot(val_acc, label='Validation Accuracy',color='royalblue')

plt.annotate('%0.3f' % acc[-1], xy=((len(acc)-2)/(len(acc)-1), acc[-1]), xytext=(1, 0.9), 
             xycoords=('axes fraction', 'data'), textcoords='axes fraction',
             arrowprops=dict(facecolor='orange',color='orange'))
plt.annotate('%0.3f' % val_acc[-1], xy=((len(acc)-2)/(len(acc)-1), val_acc[-1]), xytext=(1, 0.6), 
             xycoords=('axes fraction', 'data'), textcoords='axes fraction',
             arrowprops=dict(facecolor='royalblue',color='royalblue'))

plt.annotate('%0.3f' % acc[initial_epochs-1], xy=((initial_epochs-1)/(len(acc)-1), acc[initial_epochs-1]), xytext=(0.5, 0.5), 
             xycoords=('axes fraction', 'data'), textcoords='axes fraction',
             arrowprops=dict(facecolor='orange',color='orange'))
plt.annotate('%0.3f' % val_acc[initial_epochs-1], xy=((initial_epochs-1)/(len(acc)-1), val_acc[initial_epochs-1]), xytext=(0.3, 0.5), 
             xycoords=('axes fraction', 'data'), textcoords='axes fraction',
             arrowprops=dict(facecolor='royalblue',color='royalblue'))

plt.ylim([min(0.5,min(plt.ylim())),1.1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning',color='green')

plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss',color='orange')
plt.plot(val_loss, label='Validation Loss',color='royalblue')



plt.annotate('%0.3f' % loss[-1], xy=((len(loss)-2)/(len(loss)-1), loss[-1]), xytext=(1, 0.5), 
             xycoords=('axes fraction', 'data'), textcoords='axes fraction',
             arrowprops=dict(facecolor='orange',color='orange'))
plt.annotate('%0.3f' % val_loss[-1], xy=((len(val_loss)-2)/(len(val_loss)-1), val_loss[-1]), xytext=(1, 0.3), 
             xycoords=('axes fraction', 'data'), textcoords='axes fraction',
             arrowprops=dict(facecolor='royalblue',color='royalblue'))

plt.annotate('%0.3f' % loss[initial_epochs-1], xy=((initial_epochs-1)/(len(loss)-1), loss[initial_epochs-1]), xytext=(0.5, 0.5), 
             xycoords=('axes fraction', 'data'), textcoords='axes fraction',
             arrowprops=dict(facecolor='grey',color='orange'))
plt.annotate('%0.3f' % val_loss[initial_epochs-1], xy=((initial_epochs-1)/(len(val_loss)-1), val_loss[initial_epochs-1]), xytext=(0.3, 0.5), 
             xycoords=('axes fraction', 'data'), textcoords='axes fraction',
             arrowprops=dict(facecolor='royalblue',color='royalblue'))

plt.ylim([0, max(plt.ylim())])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

plt.savefig('Figures/{}_{}_{}_fine_tunning.png'.format(model_name,data_sets,opt_idx))
plt.show()


# Save keras model
model.save('{}_{}_{}.h5'.format(model_name,data_sets,opt_idx))
loaded_model = keras.models.load_model('{}_{}_{}.h5'.format(model_name,data_sets,opt_idx))
loss1,accuracy1= loaded_model.evaluate(test_batches)







