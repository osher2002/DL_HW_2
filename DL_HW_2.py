import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score , recall_score
from sklearn.metrics import confusion_matrix, precision_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, DenseNet121

image_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
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

N = 50;

train_data = image_generator.flow_from_directory(train_directory, 
batch_size=8, shuffle=True, class_mode='binary', target_size=(N, N))

valid_data = image_generator.flow_from_directory(valid_directory, 
batch_size=1, shuffle=False, class_mode='binary', target_size=(N,N))

test_data = image_generator.flow_from_directory(test_directory, 
batch_size=1, shuffle=False, class_mode='binary',target_size=(N, N))

type0_weigth = pneumonia_size / (normal_size + pneumonia_size);
type1_weigth = normal_size / (normal_size + pneumonia_size);

class_weight = {0: type0_weigth, 1: type1_weigth}

##ResNet Based Model

base_model = ResNet50(input_shape=(N,N,3), include_top=False, weights='imagenet')

model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128,activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1,activation="sigmoid")
])

for layer_ in base_model.layers:
    layer_.trainable = False
    
index_of_train = [87,170];

base_model.layers[index_of_train[1]].trainable = True


model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.0005),
loss = 'binary_crossentropy',
metrics=['accuracy'])

### NUMBER OF EPOCHES !
epochnum = 50;
epochs_ = range(1,epochnum+1)

history = model.fit(train_data, epochs=epochnum, validation_data=valid_data, 
class_weight=class_weight, steps_per_epoch=100, validation_steps=100)

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

### DenseNet Based Model

base_model = DenseNet121(input_shape=(N,N,3), include_top=False, weights='imagenet')

model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128,activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1,activation="sigmoid")
])

for layer_ in base_model.layers:
    layer_.trainable = False
    
index_of_train = [213,420];

base_model.layers[index_of_train[0]].trainable = True


model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.0005),
loss = 'binary_crossentropy',
metrics=['accuracy'])

### NUMBER OF EPOCHES !
epochnum = 50;
epochs_ = range(1,epochnum+1)

history = model.fit(train_data, epochs=epochnum, validation_data=valid_data, 
class_weight=class_weight, steps_per_epoch=100, validation_steps=100)

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





