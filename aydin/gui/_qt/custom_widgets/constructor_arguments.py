import inspect
import json
import docstring_parser
import numpy
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QGridLayout, QLabel, QCheckBox, QLineEdit

from aydin.util.log.log import lprint


class ConstructorArgumentsWidget(QWidget):
    def __init__(
        self,
        parent,
        arg_names=None,
        arg_defaults=None,
        arg_annotations=None,
        reference_class=None,
        disable_spatial_features=False,
    ):
        super(ConstructorArgumentsWidget, self).__init__(parent)
        self.parent = parent

        self.arg_names = []
        self.annotations = []
        self.reference_class = reference_class
        self.line_edits = []

        self.arguments_layout = QGridLayout()
        self.arguments_layout.setAlignment(Qt.AlignTop)

        for index, (name, default_value) in enumerate(zip(arg_names, arg_defaults)):

            # First we get the description:
            if inspect.isclass(reference_class):
                doc = docstring_parser.parse(reference_class.__init__.__doc__)
            else:
                doc = docstring_parser.parse(reference_class.__doc__)
            description = ""
            for param in doc.params:
                if param.arg_name == name:
                    description = param.description

            if description is not None:
                # Parameters that are marked with (hidden) in their docstrings are 'hidden:
                if '(hidden)' in description:
                    # Skip this parameter
                    continue

                # Handle None to auto replacement:
                description = description.replace("None", "'auto'")
                # Replace new lines with spaces to avoid wrapping conflicts:
                description = description.replace('\n', ' ')

            param_name = name.strip().replace('_', ' ')
            param_label = QLabel(f"{param_name}: ")
            param_label.setWordWrap(True)
            param_label.setToolTip(f"{param_name}")

            if default_value is None:
                default_value = "auto"

            if "'bool'" in str(arg_annotations[name]):
                param_edit = QCheckBox()
                param_edit.setChecked(default_value)
            elif "dtype" in str(arg_annotations[name]):
                param_edit = QLineEdit(str(default_value.__name__), self)
            else:
                param_edit = QLineEdit(str(default_value), self)

            param_edit.setToolTip(
                f"{self.annotation_prettifier(arg_annotations[name])}"
            )

            param_description = QLabel(description)
            param_description.setWordWrap(True)
            param_description.setTextInteractionFlags(Qt.TextSelectableByMouse)
            param_description.setToolTip(f"{description}")

            self.arg_names.append(name)
            self.line_edits.append(param_edit)
            self.annotations.append(arg_annotations[name])

            param_label.setFixedWidth(200)
            param_edit.setFixedWidth(100)

            if disable_spatial_features and param_name == "include spatial features":
                if "'bool'" in str(arg_annotations[name]):
                    param_edit.setChecked(False)
                else:
                    param_edit.clear()

                param_label.setEnabled(False)
                param_edit.setEnabled(False)
                param_description.setEnabled(False)

            self.arguments_layout.addWidget(
                param_label, index, 0, alignment=Qt.AlignTop
            )
            self.arguments_layout.addWidget(param_edit, index, 1, alignment=Qt.AlignTop)
            self.arguments_layout.addWidget(
                param_description, index, 2, alignment=Qt.AlignTop
            )

        if not arg_names:
            lprint("No parameters to configure")
            self.arguments_layout.addWidget(QLabel("No parameters to configure"))

        self.setLayout(self.arguments_layout)

    @staticmethod
    def annotation_prettifier(annotation):
        if "'int'" in str(annotation):
            return "int"
        elif "'float'" in str(annotation):
            return "float"
        elif "'bool'" in str(annotation):
            return "bool"
        elif "'str'" in str(annotation):
            return "str"
        else:
            response = str(annotation).replace("typing.", "")
            return response.replace("NoneType", "None")

    @property
    def params_dict(self):
        params_dict = {"class": self.reference_class, "kwargs": {}}

        for name, lineedit, annotation in zip(
            self.arg_names, self.line_edits, self.annotations
        ):

            if lineedit.text() == "auto":
                value = None
            elif "'int'" in str(annotation):
                value = int(lineedit.text())
            elif "'float'" in str(annotation):
                value = float(lineedit.text())
            elif "'bool'" in str(annotation):
                value = bool(lineedit.isChecked())
            elif "'str'" in str(annotation):
                value = lineedit.text()
            elif "dtype" in str(annotation):
                value = numpy.dtype(lineedit.text())
            elif "str" in str(annotation):
                value = lineedit.text()
            else:
                value = json.loads(lineedit.text())

            params_dict["kwargs"][name] = value

        return params_dict

    def set_advanced_enabled(self, enable: bool = False):
        for _ in range(self.arguments_layout.rowCount()):

            item = self.arguments_layout.itemAtPosition(_, 2)
            if item is not None and ("(advanced)" in item.widget().text()):
                for column_index in range(3):
                    self.arguments_layout.itemAtPosition(
                        _, column_index
                    ).widget().setHidden(not enable)
