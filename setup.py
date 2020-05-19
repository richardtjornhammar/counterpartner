import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup (
    name = "counterpartner",
    version = "0.10.1",
    author = "Richard Tjörnhammar",
    author_email = "richard.tjornhammar@gmail.com",
    description = "Counterpartner",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/richardtjornhammar/counterpartner",
    packages = setuptools.find_packages('src'),
    package_dir = {'counterpartner':'src/counterpartner','match':'src/match'},
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ] ,
)
