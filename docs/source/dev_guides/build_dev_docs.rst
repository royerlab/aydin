====================
Build dev Docs
====================

Aydin documentation webpage(aydin.app) hosts documentation only for released versions.
One can build the documentation on latest main branch commit or on a feature branch with
the following steps:

Install documentation dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can change to do `docs` folder and install documentation dependencies with
the help of `requirements-docs.txt` file by following two lines.

.. code-block:: bash

   $ cd docs
   $ pip install -r requirements-docs.txt


Clean the docs/build folder and build the docs from scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   $ make clean
   $ make html


After execution of `make html` command, one can find the built docs in the `docs/build/html`
folder. `index.html` in the mentioned folder will be the entry to point to the documentation.


Build docs to publish
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   $ make clean
   $ make publish


Our Makefile implements the `publish` command, to build documentation of all tagged versions
at the current branch. You might wondering, do we have to build docs for all versions
whenever we want to publish an update? Answer is yes. This is due to the nature of sphinx link
injection architecture and if we don't build for all versions when we have a new version
basically we will not have links for forward traverse between documentation versions.






TODO: Add how to update deployed docs on GitHub Pages.


