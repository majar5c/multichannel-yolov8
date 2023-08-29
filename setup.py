from setuptools import find_packages, setup
import pkg_resources as pkg
from pathlib import Path

FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory

REQUIREMENTS = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements((PARENT / 'requirements.txt').read_text())]

setup(name='multichannel-yolov8',
      version='1.0',
      description='Allow segmentation model training with yolov8 on NOT just RGB images',
      author='Jason Chen',
      author_email='majar62527@gmail.com',
      url='https://github.com/majar5c/mutlichannel-yolov8/',
      packages=find_packages(),
      install_requires=REQUIREMENTS,
      )