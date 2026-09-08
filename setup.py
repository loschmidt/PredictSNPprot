from setuptools import setup, find_packages

setup(
    name="predictsnp-prot-predictor",
    version="2.0.0.1.dev0",
    description="Python package for predictsnp-prot tool",
    author="Loschmidt laboratories",
    author_email="ll-admin@sci.muni.cz",
    url="https://git.loschmidt.cz/predictsnp/predictsnp-prot-predictor",
    license="MIT",
    python_requires=">=3.12",
    packages=find_packages(),
    py_modules=[],
    include_package_data=True,
    install_requires=[
        "numpy>=2.4.5",
        "scikit-learn>=1.8.0",
        "pandas>=3.0.3",
        "xgboost>=3.2.0",
        "SQLAlchemy>=2.0",
        "tqdm",
        "Bio",
        "matplotlib"
    ],
    entry_points={
        "console_scripts": [
            "predict-score=predictsnp_prot_predictor.predict_scores:main",
            "data-preparation=predictsnp_prot_predictor.training.data_preparation:main",
            "train-classifier=predictsnp_prot_predictor.training.train_classifier:main",
            "train-imputer=predictsnp_prot_predictor.training.train_imputer:main",
            "train-scaler=predictsnp_prot_predictor.training.train_scaler:main",
            "convert-artifact=predictsnp_prot_predictor.training.convert_artifact:main",
            "evaluate=predictsnp_prot_predictor.training.evaluate:main"
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: Console",
    ]
)

