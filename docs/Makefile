
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
    LATEST_VERSION = $(shell git for-each-ref --sort=taggerdate --format '%(tag)' | grep -E '^v\d+\.\d+\.\d+$$' | tail -1)
endif
ifeq ($(OS), Linux)
    LATEST_VERSION = $(shell git for-each-ref --sort=taggerdate --format '%(tag)' | grep -P '^v\d+\.\d+\.\d+$$' | tail -1)
endif

.PHONY: help clean build build-multiversion publish

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"

clean:
	-rm -rf $(BUILDDIR)/*

build:
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
