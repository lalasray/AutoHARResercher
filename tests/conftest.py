[project]
name = "autoharresearcher"
version = "0.1.0"
description = "Minimal multi-agent auto-research prototype for IMU-based HAR."
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "openai>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
