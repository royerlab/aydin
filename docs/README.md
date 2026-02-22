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

After running ``make docs`` from the project root (or ``make build`` from the ``docs/`` directory),
the generated HTML documentation can be found in the ``docs/build/html`` directory.
Open ``docs/build/html/index.html`` to view the home page for the documentation.
