"""Widget for configuring a single denoising method in the Denoise tab."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from aydin.gui._qt.custom_widgets.constructor_arguments import (
    ConstructorArgumentsWidget,
)
from aydin.gui._qt.custom_widgets.vertical_line_break_widget import (
    QVerticalLineBreakWidget,
)
from aydin.restoration.denoise.util.denoise_utils import get_denoiser_class_instance


class DenoiseTabMethodWidget(QWidget):
    """Widget displaying configuration options for a single denoising method.

    Shows a description panel on the left and configurable constructor
    arguments on the right, along with checkboxes for saving JSON options
    and the trained model.

    Parameters
    ----------
    parent : DenoiseTab
        The parent denoise tab widget.
    name : str, optional
        Backend name (e.g. 'Noise2SelfFGR-cb').
    description : str, optional
        HTML description text for the denoising method.
    disable_spatial_features : bool, optional
        If True, disables spatial feature parameters. Default is False.
    """

    def __init__(
        self, parent, name=None, description=None, disable_spatial_features=False
    ):
        super(DenoiseTabMethodWidget, self).__init__(parent)

        self.parent = parent
        self.name = name
        self.description = description

        # Widget layout
        self.main_layout = QHBoxLayout()
        self.tab_method_layout = QVBoxLayout()
        self.tab_method_layout.setAlignment(Qt.AlignTop)

        # Description Label
        self.description_scroll = QScrollArea()
        self.description_scroll.setStyleSheet("QScrollArea {border: none;}")
        self.description_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.description_scroll.setAlignment(Qt.AlignTop)
        self.description_label = QLabel(self.description)
        self.description_label.setWordWrap(True)

        self.description_label.setTextFormat(Qt.RichText)
        self.description_label.setOpenExternalLinks(True)

        self.description_label.setAlignment(Qt.AlignTop)
        self.description_scroll.setWidget(self.description_label)
        self.description_scroll.setWidgetResizable(True)
        self.description_scroll.setMinimumHeight(300)

        self.tab_method_layout.addWidget(self.description_scroll)

        self.right_side_vlayout = QVBoxLayout()
        self.right_side_vlayout.setAlignment(Qt.AlignTop)

        # Checkboxes
        self.save_json_and_model_layout = QHBoxLayout()
        self.save_json_and_model_layout.setAlignment(Qt.AlignLeft)

        self.save_json_checkbox = QCheckBox("Save denoising options (JSON)")
        self.save_json_checkbox.setChecked(True)
        self.save_json_and_model_layout.addWidget(self.save_json_checkbox)
        self.save_json_and_model_layout.addWidget(QVerticalLineBreakWidget(self))

        self.save_model_checkbox = QCheckBox("Save the trained model")
        self.save_model_checkbox.setChecked(True)
        self.save_json_and_model_layout.addWidget(self.save_model_checkbox)

        self.right_side_vlayout.addLayout(self.save_json_and_model_layout)

        # Arguments
        self.scroll = QScrollArea()
        self.scroll.setStyleSheet("QScrollArea {border: none;}")
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table_and_panes_widget = QWidget()
        self.table_and_panes_layout = QVBoxLayout()
        self.table_and_panes_widget.setLayout(self.table_and_panes_layout)
        self.scroll.setWidget(self.table_and_panes_widget)
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(300)

        self.method_and_approach, self.implementation = self.name.split("-")

        args = get_denoiser_class_instance(variant=self.name).configurable_arguments[
            self.method_and_approach + "-" + self.implementation
        ]

        self.constructor_arguments_widget_dict = {}

        for component in list(args.keys()):
            sub_dict = args[component]

            constructor_arguments_widget = ConstructorArgumentsWidget(
                self,
                arg_names=sub_dict["arguments"],
                arg_defaults=sub_dict["defaults"],
                arg_annotations=sub_dict["annotations"],
                reference_class=sub_dict["reference_class"],
                disable_spatial_features=disable_spatial_features,
            )
            self.constructor_arguments_widget_dict[component] = (
                constructor_arguments_widget
            )
            self.table_and_panes_layout.addWidget(constructor_arguments_widget)
            self.table_and_panes_layout.setSpacing(0)
            self.table_and_panes_layout.setAlignment(Qt.AlignTop)

        self.right_side_vlayout.addWidget(self.scroll)

        self.main_layout.addLayout(self.tab_method_layout, 35)
        self.main_layout.addWidget(QVerticalLineBreakWidget(self))
        self.main_layout.addLayout(self.right_side_vlayout, 50)

        self.setLayout(self.main_layout)

    def lower_level_args(self):
        """Collect all constructor arguments for this denoising method.

        Returns
        -------
        dict
            Dictionary mapping component names to their parameter
            dictionaries, plus a 'variant' key with the method name.
        """
        args = {}

        for key, value in self.constructor_arguments_widget_dict.items():
            args[key] = value.params_dict

        args["variant"] = self.name

        return args
