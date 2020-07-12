# Augmented_Reality_with_OpenCV
Augmented Reality using OpenCV and Python

![Demo Result](https://github.com/kranok-dev/Augmented_Reality_with_OpenCV/blob/master/Result.jpg?raw=true)

**Description**                                                               
> This application consists on doing Augmented Reality using Python and OpenCV. It only requires images as references, OBJ files and either a live video feed or an image with the references. 

> This work is based on Juan Gallostra's code:
> https://github.com/juangallostra/augmented-reality

> The 3D-Models (OBJ Files) were obtained from https://clara.io/library. There are plenty more models of different objects.

**Installation**
> Clone this repository and the implemented code requires OpenCV to be installed in Python (Python 3 was used):
  ```
  $ sudo apt-get install build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev libhdf5-103 libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5 python3-dev
  
  $ sudo pip3 install opencv-contrib-python
  ```

**Execution**
> The application can be executed either using a webcam (already integrated on computer or USB), or only processing saved images:
```
$ sudo python3 app_video.py

$ sudo python3 app_image.py
```

> Try the demos implemented, change the models, download other 3D-Models and have fun!

**Demo Video**
>
