import setuptools

requirements = []
with open('requirements.txt', 'rt') as f:
    for req in f.read().splitlines():
        if req.startswith('git+'):
            pkg_name = req.split('/')[-1].replace('.git', '')
            requirements.append(f'{pkg_name} @ {req}')
        else:
            requirements.append(req)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cedr",
    version="0.0.1",
    author="Sean MacAvaney",
    author_email="sean@ir.cs.georgetown.edu",
    description="Code for CEDR: Contextualized Embeddings for Document Ranking, at SIGIR 2019.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Georgetown-IR-Lab/cedr",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.6',
)
