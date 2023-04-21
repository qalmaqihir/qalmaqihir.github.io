Tensorflow BootCamp - Advance Tensorflow Usage
================
by Jawad Haider

- <a href="#tpu" id="toc-tpu">TPU</a>

## TPU

``` python
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, \
  GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
```

``` python
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
```

    INFO:tensorflow:Deallocate tpu buffers before initializing tpu system.

    INFO:tensorflow:Deallocate tpu buffers before initializing tpu system.

    INFO:tensorflow:Initializing the TPU system: grpc://10.117.37.210:8470

    INFO:tensorflow:Initializing the TPU system: grpc://10.117.37.210:8470

    INFO:tensorflow:Finished initializing TPU system.

    INFO:tensorflow:Finished initializing TPU system.

    All devices:  [LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:0', device_type='TPU'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:1', device_type='TPU'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:2', device_type='TPU'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:3', device_type='TPU'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:4', device_type='TPU'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:5', device_type='TPU'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:6', device_type='TPU'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:7', device_type='TPU')]

``` python
strategy = tf.distribute.TPUStrategy(resolver)
```

    INFO:tensorflow:Found TPU system:

    INFO:tensorflow:Found TPU system:

    INFO:tensorflow:*** Num TPU Cores: 8

    INFO:tensorflow:*** Num TPU Cores: 8

    INFO:tensorflow:*** Num TPU Workers: 1

    INFO:tensorflow:*** Num TPU Workers: 1

    INFO:tensorflow:*** Num TPU Cores Per Worker: 8

    INFO:tensorflow:*** Num TPU Cores Per Worker: 8

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:1, TPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:1, TPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:2, TPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:2, TPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:3, TPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:3, TPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:4, TPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:4, TPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:5, TPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:5, TPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)

    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)

``` python
# Note: model creation must be in strategy scope
# We will define the function now, but this code
# won't run outside the scope
```

``` python
def create_model():
  i = Input(shape=(32, 32, 3))

  x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
  x = BatchNormalization()(x)
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D((2, 2))(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D((2, 2))(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D((2, 2))(x)

  x = Flatten()(x)
  x = Dropout(0.2)(x)
  x = Dense(1024, activation='relu')(x)
  x = Dropout(0.2)(x)
  x = Dense(10)(x)

  model = Model(i, x)
  return model
```

``` python
# Load in the data
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()
print("x_train.shape:", x_train.shape)
print("y_train.shape", y_train.shape)
```

    Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    170500096/170498071 [==============================] - 2s 0us/step
    170508288/170498071 [==============================] - 2s 0us/step
    x_train.shape: (50000, 32, 32, 3)
    y_train.shape (50000,)

``` python
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
```

``` python
with strategy.scope():
  model = create_model()
  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['sparse_categorical_accuracy'])

batch_size = 256

# reshuffle_each_iteration=None is default but is later set to True if None
# thus "True" is the actual default
train_dataset = train_dataset.shuffle(1000).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

model.fit(
    train_dataset,
    epochs=5,
    validation_data=test_dataset)
```

    Epoch 1/5
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/keras/engine/training.py:2970: StrategyBase.unwrap (from tensorflow.python.distribute.distribute_lib) is deprecated and will be removed in a future version.
    Instructions for updating:
    use `experimental_local_results` instead.

    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/keras/engine/training.py:2970: StrategyBase.unwrap (from tensorflow.python.distribute.distribute_lib) is deprecated and will be removed in a future version.
    Instructions for updating:
    use `experimental_local_results` instead.

    196/196 [==============================] - 23s 73ms/step - loss: 1.6891 - sparse_categorical_accuracy: 0.4619 - val_loss: 4.2676 - val_sparse_categorical_accuracy: 0.1514
    Epoch 2/5
    196/196 [==============================] - 9s 46ms/step - loss: 1.0112 - sparse_categorical_accuracy: 0.6388 - val_loss: 2.1024 - val_sparse_categorical_accuracy: 0.3468
    Epoch 3/5
    196/196 [==============================] - 9s 44ms/step - loss: 0.7856 - sparse_categorical_accuracy: 0.7238 - val_loss: 0.8410 - val_sparse_categorical_accuracy: 0.7061
    Epoch 4/5
    196/196 [==============================] - 9s 45ms/step - loss: 0.6428 - sparse_categorical_accuracy: 0.7735 - val_loss: 0.6944 - val_sparse_categorical_accuracy: 0.7592
    Epoch 5/5
    196/196 [==============================] - 9s 46ms/step - loss: 0.5314 - sparse_categorical_accuracy: 0.8134 - val_loss: 0.6975 - val_sparse_categorical_accuracy: 0.7645

    <keras.callbacks.History at 0x7ff8093771d0>

``` python
model.save('mymodel.h5')
```

``` python
with strategy.scope():
  model = tf.keras.models.load_model('mymodel.h5')
  out = model.predict(x_test[:1])
  print(out)
```

    [[-1.595037   -2.7977936   0.59842265  6.563197   -1.4122396   4.490996
       4.286044   -2.2540898  -3.8138728  -2.6489956 ]]

``` python
# Note: old bug
# https://www.kaggle.com/c/flower-classification-with-tpus/discussion/148615
```

<center>

<a href=''> ![Logo](../logo1.png) </a>

</center>
<center>
<em>Copyright Qalmaqihir</em>
</center>
<center>
<em>For more information, visit us at
<a href='http://www.github.com/qalmaqihir/'>www.github.com/qalmaqihir/</a></em>
</center>
