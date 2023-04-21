Tensorflow BootCamp - Colab Basics Installing Tensorflow
================
by Jawad Haider

``` python
# What's already installed?
# import tensorflow as tf
# print(tf.__version__)
```

    1.14.0

``` python
# Install TensorFlow 2.0
# You can run regular shell commands by prepending !
# !pip install -q tensorflow==2.0.0-beta1

# GPU version
# !pip install -q tensorflow-gpu==2.0.0-beta1
```

``` python
##### UPDATE 2020 #####
# new feature of colab - you can just use this
try:
  %tensorflow_version 2.x  # Colab only.
except Exception:
  pass
```

    `%tensorflow_version` only switches the major version: 1.x or 2.x.
    You set: `2.x  # Colab only.`. This will be interpreted as: `2.x`.


    TensorFlow 2.x selected.

``` python
# Check Tensorflow version again
import tensorflow as tf
print(tf.__version__)
```

    2.2.0-rc2

``` python
# How to install a library permanently?
# https://stackoverflow.com/questions/55253498/how-do-i-install-a-library-permanently-in-colab
```

``` python
# More fun with !
!ls
```

    sample_data

``` python
# More fun with !
# Nice! Looks like we already have some useful data to work with
!ls sample_data
```

    anscombe.json             mnist_test.csv
    california_housing_test.csv   mnist_train_small.csv
    california_housing_train.csv  README.md

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

``` python
s
```
