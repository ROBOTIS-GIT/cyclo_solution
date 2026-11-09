from glob import glob

from setuptools import find_packages, setup

package_name = 'ffw_vision'
setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=[]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/web', glob('web/*')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/models', glob('models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Jiho Park',
    author_email='jiho010920@gmail.com',
    maintainer='Jiho Park',
    maintainer_email='jiho010920@gmail.com',
    description='AI Worker vision nodes that publish pixel observations.',
    license='Apache 2.0',
    entry_points={
        'console_scripts': [
            'depth_probe = ffw_vision.depth_probe_node:main',
            'input_monitor = ffw_vision.input_monitor_node:main',
            'object_detector = ffw_vision.object_detector_node:main',
            'pixel_observer = ffw_vision.pixel_observer_node:main',
            'web_select = ffw_vision.web_select_node:main',
        ],
    },
)
