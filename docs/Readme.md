# Small tutorial for creating docs (for internal reference...)

Setting up the environment:

```
python3.10 -m venv docs_venv
source docs_venv/bin/activate
sphinx-quickstart
```

An interactive terminal will ask you to manually configure several options for your sphinx build.

If you want to use the "Read the Docs" theme, install it with `pip`:

```
pip install sphinx-rtd-theme
```

Then, uncomment the first three lines in the `source/conf.py` file, and modify the `sys.path.insert` path to point to your sources' base path. The lines should look like:

```
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
```

If, as in our case, you use Google-style docstrings in your code, add the following extension in the `source/conf.py`:

```
extensions = ["sphinx.ext.napoleon"]
```

And, for setting the "Read the Docs" theme, set the `html_theme` variable to:

```
html_theme = 'sphinx_rtd_theme'
```

Auto-document the modules (out of the docstrings):

```
mkdir source/modules
sphinx-apidoc -f -o source/modules ../heightmap_interpolation/
```

In our case, we want some other specific configuration options:

```
sphinx-apidoc -fMe -H "Python Packages and Modules" -o source/modules ../heightmap_interpolation/
```

Also, if your package depends on several other modules, and sphinx-apidoc complains about missing modules, then install the requirements of your project within the current virtual environment also. In our case (from this folder):

```
pip install -r ../requirements.txt
```

Finally compile using the Sphinx-generated makefile:

```
make html
```

