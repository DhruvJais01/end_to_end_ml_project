from typing import List
from setuptools import find_packages, setup


def get_requirements(file_path: str) -> List[str]:
    """This function returns the list of requirements from the requirements.txt file."""
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements


setup(
    name="end_to_end_ml_project",
    version="0.0.1",
    author="Dhruv",
    author_email="dhruvjaiswal400@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    description="MLflow project",
)
