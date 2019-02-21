import setuptools

setuptools.setup(
    author="Seaton Ullberg",
    author_email="sullberg@ufl.edu",
    entry_points="""
        [console_scripts]
        grains=grains.cli:cli
    """,
    include_package_data=True,
    install_requires=[
        "Click",
    ],
    name="grains",
    packages=["grains"],
    url="https://github.com/seatonullberg/grains",
    version="0.1.0",
)