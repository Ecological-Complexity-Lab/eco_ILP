from setuptools import setup, find_packages

setup(
    name="ecoILP",
    version="1.0.0",
    author="ecomplab",
    # author_email="email@example.com",
    description="A package for ecological link prediction using machine learning.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ecological-Complexity-Lab/eco_ILP",
    packages=find_packages(),  # Automatically find all packages
    package_data={
        "ecoILP": ["models/*", "helper/*"],  # Include all files in the models/ directory
    },
    include_package_data=True,  # Include non-Python files like models/
    install_requires=[
        "joblib==1.2.0",
        "numpy==1.22.4",
        "pandas==1.5.3",
        "matplotlib==3.7.0",
        "networkx==3.0",
        "seaborn==0.12.2",
        "scikit-learn==1.2.1",
        "tqdm==4.65.0",
        "pyyaml==6.0",
        "gradio==4.44.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        # "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
