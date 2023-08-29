from distutils.core import setup

setup(name='multichannel-yolov8',
      version='1.0',
      description='Allow segmentation model training with yolov8 on NOT just RGB images',
      author='Jason Chen',
      author_email='majar62527@gmail.com',
      url='https://github.com/majar5c/mutlichannel-yolov8/',
      packages=['multichannel_yolov8', 'multichannel_yolov8.utils', 'multichannel_yolov8.model'],
      )