+from setuptools import setup
import goic


long_desc = ''

setup(
    name="goic",
    description="""A Minimal Gabor-based Image Classifier""",
    long_description=long_desc,
    version=microtc.__version__,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        'Programming Language :: Python :: 3',
        "Topic :: Scientific/Engineering :: Artificial Intelligence"],

    packages=['goic', 'goic/tests', 'goic/tools'],
    include_package_data=True,
    zip_safe=False,
    package_data={
        'microtc/tests': ['text.json'],
    },
    scripts=[
        'goic/tools/goic-train',
        'goic/tools/goic-predict',
        'goic/tools/goic-params',
        'goic/tools/goic-model',
        'goic/tools/goic-perf',
        'goic/tools/goic-ensemble',
        'goic/tools/goic-kfolds'
    ]
)
