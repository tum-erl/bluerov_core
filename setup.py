import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'bluerov_core'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Προσθήκες για τα launch και urdf αρχεία:
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'bluerov_system = bluerov_core.bluerov_system_node:main',
            'localization = bluerov_core.localization_node:main',
            'localization2 = bluerov_core.localization_node2:main',
            'camera_publisher = bluerov_core.camera_publisher_node:main',
            'camera_publisher2 = bluerov_core.camera_publisher2:main',
            'laptop_cam_publisher = bluerov_core.laptop_camera_publisher_node:main'
        ],
    },
)