# Copyright (c) 2021 Massachusetts Institute of Technology

from setuptools import find_packages, setup

DISTNAME = "hydra_zen_example.image_classifier"
LICENSE = "MIT"
INSTALL_REQUIRES = [
    "hydra-zen >= 0.3.0rc5",
    "pytorch_lightning >= 1.4.9",
    "pytorch >= 1.7.1",
    "torchvision >= 0.8.1",
    "torchmetrics >= 0.5.1",
]
TESTS_REQUIRE = [
    "pytest >= 3.8",
]


setup(
    name=DISTNAME,
    license=LICENSE,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    python_requires=">=3.7",
    packages=["hydra_zen_example.image_classifier"],
)
