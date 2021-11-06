# Aydin documentation

Aydin documentation dependencies can be installed as shown(assuming you are inside `aydin/docs` folder already):

```shell script
# Create a new environment
conda create -y -n aydindocs python=3.9

# Activate the environment
conda activate aydindocs

# Install Aydin
pip install -e ../.

# Install Aydin documentation dependencies
pip install -r requirements-docs.txt
```

After running ``make html`` the generated HTML documentation can be found in
the ``build/html`` directory. Open ``build/html/index.html`` to view the home
page for the documentation.
