from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QVBoxLayout,
    QScrollArea,
)

from aydin.gui._qt.custom_widgets.constructor_arguments import (
    ConstructorArgumentsWidget,
)
from aydin.gui._qt.job_runners.preview_job_runner import PreviewJobRunner
from aydin.gui._qt.custom_widgets.horizontal_line_break_widget import (
    QHorizontalLineBreakWidget,
)
from aydin.gui._qt.custom_widgets.vertical_line_break_widget import (
    QVerticalLineBreakWidget,
)
from aydin.util.string.break_text import break_text


class TransformsTabItem(QWidget):
    def __init__(
        self,
        parent,
        name=None,
        arg_names=None,
        arg_defaults=None,
        arg_annotations=None,
        transform_class=None,
    ):
        super(TransformsTabItem, self).__init__(parent)

        self.parent = parent
        self.arg_names = arg_names
        self.annotations = []
        self.transform_class = transform_class
        self.line_edits = []

        self.layout = QHBoxLayout()

        explanation_text_string = self.transform_class.__doc__
        explanation_text_string = break_text(explanation_text_string)
        explanation_text_string = explanation_text_string.replace('\n', '<br>')
        self.explanation_text = QLabel(explanation_text_string, self)
        self.explanation_text.setTextFormat(Qt.RichText)
        self.explanation_text.setOpenExternalLinks(True)
        self.explanation_text.setAlignment(Qt.AlignTop)
        self.layout.addWidget(self.explanation_text, 45)

        # Vertical Line Break
        self.layout.addWidget(QVerticalLineBreakWidget(self))

        self.transform_details_layout = QVBoxLayout()
        self.transform_details_layout.setAlignment(Qt.AlignTop)

        # Checkbox
        self.enabling_checkboxes_layout = QHBoxLayout()

        self.preprocess_checkbox = QCheckBox("Enable" if name is None else name)
        self.enabling_checkboxes_layout.addWidget(self.preprocess_checkbox)
        self.enabling_checkboxes_layout.addStretch()

        self.preview_job_runner = PreviewJobRunner(
            self, self.parent.parent.parent.threadpool
        )
        self.enabling_checkboxes_layout.addWidget(self.preview_job_runner)

        self.transform_details_layout.addLayout(self.enabling_checkboxes_layout)

        # Postprocess checkbox
        self.postprocess_checkbox = QCheckBox(
            self.transform_class.postprocess_description
        )
        self.postprocess_checkbox.setEnabled(
            self.transform_class.postprocess_supported
            and self.preprocess_checkbox.isChecked()
        )
        self.postprocess_checkbox.setChecked(
            self.transform_class.postprocess_recommended
        )
        self.preprocess_checkbox.stateChanged.connect(
            self.preprocess_chechbox_on_state_changed
        )
        self.preprocess_chechbox_on_state_changed()
        self.transform_details_layout.addWidget(self.postprocess_checkbox)

        self.transform_details_layout.addWidget(QHorizontalLineBreakWidget(self))

        # Parameters
        self.scroll = QScrollArea()
        self.scroll.setStyleSheet("QScrollArea {border: none;}")
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.constructor_arguments_widget = ConstructorArgumentsWidget(
            self,
            arg_names=arg_names,
            arg_defaults=arg_defaults,
            arg_annotations=arg_annotations,
            reference_class=transform_class,
        )
        self.scroll.setWidget(self.constructor_arguments_widget)
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(200)
        self.transform_details_layout.addWidget(self.scroll)

        self.layout.addLayout(self.transform_details_layout, 45)
        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)

    def preprocess_chechbox_on_state_changed(self):
        self.postprocess_checkbox.setEnabled(self.preprocess_checkbox.isChecked())

    @property
    def params_dict(self):
        if self.preprocess_checkbox.isChecked():
            params_dict = self.constructor_arguments_widget.params_dict
            params_dict["kwargs"][
                "do_postprocess"
            ] = self.postprocess_checkbox.isChecked()

            return params_dict
