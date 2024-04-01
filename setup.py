from setuptools import find_packages, setup

def import_req(file_name):
    with open(file_name) as file_obj:
        data = file_obj.readlines()
        requirements = [req.replace("\n", "").replace("-e .", "") for req in data]
    return requirements

setup(
    name = "First ML Project",
    version = "0.0.1",
    author = "Sharooq Farzeen A K",
    author_email = "sharooqfarzeen@gmail.com",
    packages = find_packages(),
    install_requires = import_req('requirements.txt'),
)