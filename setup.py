import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'face_aware_3dbbox'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('lib/python3.10/site-packages', package_name, 'weights'), 
         glob('face_aware_3dbbox/weights/*')), 
        (os.path.join('lib/python3.10/site-packages', package_name, 'torch_lib'), 
         glob('face_aware_3dbbox/torch_lib/*.txt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='taewook',
    maintainer_email='utsi09@g.skku.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bbox3d_node = face_aware_3dbbox.bbox3d_node:main',
            'face_aware_3dbbox = face_aware_3dbbox.face_aware_3dbbox:main',
        ],
    },
)
