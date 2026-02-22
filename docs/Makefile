
# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
BUILDDIR      = build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

# get the most recent tagged version
# (note that the regex here must be consistent with `smv_tag_whitelist` in the sphinx config)
OS := $(shell uname)
ifeq ($(OS), Darwin)
    LATEST_VERSION = $(shell git for-each-ref --sort=taggerdate --format '%(tag)' | grep -E '^v\d+\.\d+\.\d+(\.\d+)?$$' | tail -1)
endif
ifeq ($(OS), Linux)
    LATEST_VERSION = $(shell git for-each-ref --sort=taggerdate --format '%(tag)' | grep -P '^v\d+\.\d+\.\d+(\.\d+)?$$' | tail -1)
endif

.PHONY: help clean screenshots build build-multiversion publish

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  screenshots        to regenerate napari plugin screenshots"
	@echo "  build              to make standalone HTML files (regenerates screenshots first)"
	@echo "  build-multiversion to build docs for all tagged versions"
	@echo "  publish            to build multi-version docs with index redirect"
	@echo "  clean              to remove all build output"

clean:
	-rm -rf $(BUILDDIR)/*

screenshots:
	python source/_scripts/generate_napari_screenshots.py

build: screenshots
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

build-multiversion:
	@echo ${LATEST_VERSION}
	sphinx-multiversion source build/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

index.html:
	@sed 's/LATEST_VERSION/${LATEST_VERSION}/g' index-template.html > $(BUILDDIR)/html/index.html

publish: build-multiversion index.html
