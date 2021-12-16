import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score , recall_score
from sklearn.metrics import confusion_matrix, precision_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_generator = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.08,
    shear_range=0.07,
    zoom_range=0.09,
    samplewise_center=True,
    samplewise_std_normalization=True
)

train_directory = 'data/train'
valid_directory = 'data/val'
test_directory = 'data/test'

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

type1 = 'NORMAL';
type0 = 'PNEUMONIA';

pneumonia_size = len(os.listdir(os.path.join(train_directory, type0)))
normal_size = len(os.listdir(os.path.join(train_directory, type1)))

N = 100;

train_data = image_generator.flow_from_directory(train_directory, 
batch_size=8, shuffle=True, class_mode='binary', target_size=(N, N))

valid_data = image_generator.flow_from_directory(valid_directory, 
batch_size=1, shuffle=False, class_mode='binary', target_size=(N,N))

test_data = image_generator.flow_from_directory(test_directory, 
batch_size=1, shuffle=False, class_mode='binary',target_size=(N, N))

type0_weigth = pneumonia_size / (normal_size + pneumonia_size);
type1_weigth = normal_size / (normal_size + pneumonia_size);

class_weight = {0: type0_weigth, 1: type1_weigth}

#DNN FULLY CONNECTED
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (N,N,3)),
tf.keras.layers.Dense(1000, activation=tf.nn.relu),
tf.keras.layers.Dense(250, activation=tf.nn.relu),
tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)]);

model.compile(optimizer = tf.optimizers.Adam(),
loss = 'binary_crossentropy',
metrics=['accuracy'])

### NUMBER OF EPOCHES !
epochnum = 30;
epochs_ = range(1,epochnum+1)

history = model.fit(train_data, epochs=epochnum, validation_data=valid_data, 
class_weight=class_weight, steps_per_epoch=8, validation_steps=8)

#LOSS CALC & PLOTTING
loss_train = history.history['loss']
loss_val = history.history['val_loss']

plt.plot(epochs_, loss_train, 'g', label='Training loss')
plt.plot(epochs_, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#ACCURACY PLOTTING
acc_train = history.history['accuracy']
acc_val = history.history['val_accuracy']

plt.plot(epochs_, acc_train, 'g', label='Training accuracy')
plt.plot(epochs_, acc_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

predict = model.predict(test_data);

test_predict = predict > 0.5;

test_conf_mat = confusion_matrix(test_data.classes, test_predict);

test_precision = precision_score(test_data.classes, test_predict);
test_recall = recall_score(test_data.classes, test_predict);
roc_auc_score(test_data.classes,predict);

#CNN network for Classifying

model = tf.keras.models.Sequential([
    
#The first convolution layer
tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(N, N, 3)),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.MaxPooling2D(2, 2),

# The second convolution layer
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.MaxPooling2D(2,2),

# The third convolution layer
tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.MaxPooling2D(2,2),

#Fully Connected
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(1, activation='sigmoid')]);

#Choosing Learning rate and analyze effect on results:

model.compile(loss='binary_crossentropy',
optimizer=tf.optimizers.Adam(learning_rate=0.005), metrics='accuracy')

#FITTING THE MODEL
epochnum = 30;
epochs_ = range(1,epochnum+1)

history = model.fit(train_data, epochs=epochnum, validation_data=valid_data, 
class_weight=class_weight, steps_per_epoch=100, validation_steps=25)

#LOSS CALC & PLOTTING
loss_train = history.history['loss']
loss_val = history.history['val_loss']

plt.plot(epochs_, loss_train, 'g', label='Training loss')
plt.plot(epochs_, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#ACCURACY PLOTTING
acc_train = history.history['accuracy']
acc_val = history.history['val_accuracy']

plt.plot(epochs_, acc_train, 'g', label='Training accuracy')
plt.plot(epochs_, acc_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

predict = model.predict(test_data);

test_predict = predict > 0.5;

test_conf_mat = confusion_matrix(test_data.classes, test_predict);

test_precision = precision_score(test_data.classes, test_predict);
test_recall = recall_score(test_data.classes, test_predict);
auc_score = roc_auc_score(test_data.classes,predict);












