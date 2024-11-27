
from setuptools import setup, find_packages

setup(
    name='DetectionInfer',             # 项目的名称
    version='1.0.0',               # 项目的版本号
    packages=find_packages(),      # 自动找到项目中的所有包
    include_package_data=True,     # 包含包内所有数据文件
    install_requires=[             # 定义项目依赖的第三方包
        'pyyaml',
        # "onnxruntime-gpu",
        "opencv-python",
        "pycryptodome"
    ],
    # entry_points={                 # 定义命令行工具入口点
    #     'console_scripts': [
    #         'mycli = mypackage.module:main_function',
    #     ],
    # },
    author='QirongMo',            # 项目作者
    # author_email='you@example.com',# 作者邮箱
    description='目标检测模型推理',  # 项目描述
    # long_description=open('README.md').read(),  # 项目详细描述
    # long_description_content_type='text/markdown',  # 描述内容类型
    # url='https://github.com/yourusername/myproject',  # 项目的URL
    license='MIT',                 # 项目许可证
    classifiers=[                  # 分类标签
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
