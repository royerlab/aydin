"""Tab widget for dynamically loading and configuring image transforms."""

import importlib
import inspect
import pkgutil

from qtpy.QtWidgets import QTabWidget

from aydin.gui._qt.transforms_tab_item import TransformsTabItem
from aydin.it import transforms
from aydin.util.log.log import aprint


class TransformsTabWidget(QTabWidget):
    """Tab widget that dynamically loads all available image transforms.

    Discovers transform classes from ``aydin.it.transforms``, creates a
    ``TransformsTabItem`` widget for each, and provides basic/advanced
    mode toggling.

    Parameters
    ----------
    parent : ProcessingTab
        The parent processing tab widget.
    """

    def __init__(self, parent):
        """Initialize by discovering and loading all available transform classes.

        Parameters
        ----------
        parent : ProcessingTab
            The parent processing tab widget.
        """
        super(TransformsTabWidget, self).__init__(parent)

        self.parent = parent
        self.main_page = parent.parent
        self.list_of_item_widgets = []

        for module in [
            x
            for x in pkgutil.iter_modules(tuple(transforms.__path__))
            if not x.ispkg and x.name != 'base'
        ]:
            name = module.name  # name is filename

            response = importlib.import_module(transforms.__name__ + '.' + module.name)
            candidates = [
                x for x in dir(response) if module.name.replace('_', '') in x.lower()
            ]
            if not candidates:
                aprint(f"No transform class found in module '{module.name}', skipping")
                continue
            elem = candidates[0]  # class name

            class_itself = response.__getattribute__(elem)
            fullargspec = inspect.getfullargspec(class_itself.__init__)

            widget = TransformsTabItem(
                self,
                name=class_itself.preprocess_description,
                arg_names=fullargspec.args[1:],
                arg_defaults=fullargspec.defaults,
                arg_annotations=fullargspec.annotations,
                transform_class=class_itself,
                main_page=self.main_page,
            )
            self.list_of_item_widgets.append(widget)

            self.addTab(widget, name.replace("_", " "))

            if name in ["range", "padding", "vst"]:
                widget.preprocess_checkbox.setChecked(True)

    def set_advanced_enabled(self, enable: bool = False):
        """Toggle visibility of advanced transforms and their parameters.

        Parameters
        ----------
        enable : bool, optional
            If True, show tabs for advanced transforms. Default is False.
        """
        for index, item_widget in enumerate(self.list_of_item_widgets):
            if "(advanced)" in item_widget.transform_class.__doc__:
                self.setTabVisible(index, enable)

            item_widget.constructor_arguments_widget.set_advanced_enabled(enable=enable)
