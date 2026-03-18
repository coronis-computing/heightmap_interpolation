# Small tutorial for creating docs (for internal reference...)

In order to build the docs, install the additional dependencies. From the root dir of this project:

```
pip install .[docs]
```

Then, compile using the Sphinx-generated makefile that you can find in the folder containing this readme (`./docs`):

```
make html
```

