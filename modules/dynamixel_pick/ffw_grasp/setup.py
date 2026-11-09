from glob import glob

from setuptools import find_packages, setup

package_name = 'ffw_grasp'
setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=[]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Jiho Park',
    author_email='jiho010920@gmail.com',
    maintainer='Jiho Park',
    maintainer_email='jiho010920@gmail.com',
    description='AI Worker grasp poses in base_link from vision object observations.',
    license='Apache 2.0',
    entry_points={
        'console_scripts': [
            'grasp_planner = ffw_grasp.grasp_planner_node:main',
        ],
    },
)
