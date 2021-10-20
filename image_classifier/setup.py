# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from setuptools import find_packages, setup

DISTNAME = "hydra_zen_examples"
LICENSE = "MIT"
INSTALL_REQUIRES = [
    "hydra-core >= 1.1.0",
    "typing-extensions >= 3.10.0.1",
]
TESTS_REQUIRE = [
    "pytest >= 3.8",
    "hypothesis >= 6.16.0",
]


setup(
    name=DISTNAME,
    license=LICENSE,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    python_requires=">=3.6",
    packages=["image_classifier"],
)
