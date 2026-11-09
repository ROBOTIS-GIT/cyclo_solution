from glob import glob

from setuptools import find_packages, setup

package_name = 'ffw_pick'
setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Jiho Park',
    author_email='jiho010920@gmail.com',
    maintainer='Jiho Park',
    maintainer_email='jiho010920@gmail.com',
    description='AI Worker pick with MoveIt: analytic arm IK, least-motion approach and lift.',
    license='Apache 2.0',
    entry_points={
        'console_scripts': [
            'pick_moveit = ffw_pick.pick_moveit:main',
            'pick_action_server = ffw_pick.action_server:main',
        ],
    },
)
