# Aydin documentation

Aydin documentation dependencies can be installed as shown (assuming you are inside the `docs` folder already):

```bash
# Install Aydin with documentation dependencies
pip install -e ".[docs]"
```

Or using the Makefile from the project root:

```bash
make setup  # installs dev + docs dependencies
```

After running ``make publish`` the generated HTML documentation can be found in
the ``build/html`` directory. Open ``build/html/index.html`` to view the home
page for the documentation.
