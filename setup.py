
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Optimized Kalman Filter",
    version="0.0.3",
    license='MIT',
    author="Ido Greenberg",
    description="Optimization of a Kalman Filter from data of states and their observations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url="https://github.com/ido90/Optimized-Kalman-Filter",
    keywords = ["OKF", "Optimized Kalman Filter", "Kalman Filter"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.5',
    py_modules=["okf"],
    install_requires=["torch"]
)
