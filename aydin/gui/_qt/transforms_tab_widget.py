import importlib
import inspect
import pkgutil
from qtpy.QtWidgets import QTabWidget

from aydin.gui._qt.transforms_tab_item import TransformsTabItem
from aydin.it import transforms


class TransformsTabWidget(QTabWidget):
    def __init__(self, parent):
        super(TransformsTabWidget, self).__init__(parent)

        self.parent = parent
        self.list_of_item_widgets = []

        for module in [
            x
            for x in pkgutil.iter_modules(tuple(transforms.__path__))
            if not x.ispkg and x.name != 'base'
        ]:
            name = module.name  # name is filename

            response = importlib.import_module(transforms.__name__ + '.' + module.name)
            elem = [
                x for x in dir(response) if module.name.replace('_', '') in x.lower()
            ][
                0
            ]  # class name

            class_itself = response.__getattribute__(elem)
            fullargspec = inspect.getfullargspec(class_itself.__init__)

            widget = TransformsTabItem(
                self,
                name=class_itself.preprocess_description,
                arg_names=fullargspec.args[1:],
                arg_defaults=fullargspec.defaults,
                arg_annotations=fullargspec.annotations,
                transform_class=class_itself,
            )
            self.list_of_item_widgets.append(widget)

            self.addTab(widget, name.replace("_", " "))

            if name in ["range", "padding", "vst"]:
                widget.preprocess_checkbox.setChecked(True)

    def clear_the_list(self):
        self.clear()

    def set_advanced_enabled(self, enable: bool = False):
        for index, item_widget in enumerate(self.list_of_item_widgets):
            if "(advanced)" in item_widget.transform_class.__doc__:
                self.setTabVisible(index, enable)

            item_widget.constructor_arguments_widget.set_advanced_enabled(enable=enable)
