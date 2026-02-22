"""Shared widget setup helpers for denoise tab method widgets."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout

from aydin.gui._qt.custom_widgets.vertical_line_break_widget import (
    QVerticalLineBreakWidget,
)


def setup_description_scroll(parent, description):
    """Create a scrollable description label with rich text support.

    Parameters
    ----------
    parent : QWidget
        Parent widget.
    description : str
        HTML description text.

    Returns
    -------
    description_scroll : QScrollArea
        The scroll area containing the description label.
    description_label : QLabel
        The label displaying the description.
    """
    description_scroll = QScrollArea()
    description_scroll.setStyleSheet("QScrollArea {border: none;}")
    description_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    description_scroll.setAlignment(Qt.AlignTop)
    description_label = QLabel(description)
    description_label.setWordWrap(True)
    description_label.setTextFormat(Qt.RichText)
    description_label.setOpenExternalLinks(True)
    description_label.setAlignment(Qt.AlignTop)
    description_scroll.setWidget(description_label)
    description_scroll.setWidgetResizable(True)
    description_scroll.setMinimumHeight(300)
    return description_scroll, description_label


def setup_save_checkboxes(parent):
    """Create the save JSON and save model checkbox row.

    Parameters
    ----------
    parent : QWidget
        Parent widget.

    Returns
    -------
    layout : QHBoxLayout
        Horizontal layout containing the checkboxes.
    save_json_checkbox : QCheckBox
        Checkbox for saving denoising options.
    save_model_checkbox : QCheckBox
        Checkbox for saving the trained model.
    """
    layout = QHBoxLayout()
    layout.setAlignment(Qt.AlignLeft)

    save_json_checkbox = QCheckBox("Save denoising options (JSON)")
    save_json_checkbox.setChecked(True)
    layout.addWidget(save_json_checkbox)
    layout.addWidget(QVerticalLineBreakWidget(parent))

    save_model_checkbox = QCheckBox("Save the trained model")
    save_model_checkbox.setChecked(True)
    layout.addWidget(save_model_checkbox)

    return layout, save_json_checkbox, save_model_checkbox


def setup_denoise_tab_layouts(parent, description):
    """Set up the common two-column layout for denoise method widgets.

    Creates the main horizontal layout with a description column (left)
    and a right-side column containing save checkboxes.

    Parameters
    ----------
    parent : QWidget
        Parent widget.
    description : str
        HTML description text.

    Returns
    -------
    main_layout : QHBoxLayout
        The top-level horizontal layout.
    tab_method_layout : QVBoxLayout
        Left column layout for description.
    right_side_vlayout : QVBoxLayout
        Right column layout for arguments and options.
    description_scroll : QScrollArea
        The description scroll area.
    description_label : QLabel
        The description label.
    save_json_checkbox : QCheckBox
        Save JSON options checkbox.
    save_model_checkbox : QCheckBox
        Save model checkbox.
    """
    main_layout = QHBoxLayout()
    tab_method_layout = QVBoxLayout()
    tab_method_layout.setAlignment(Qt.AlignTop)

    description_scroll, description_label = setup_description_scroll(
        parent, description
    )
    tab_method_layout.addWidget(description_scroll)

    right_side_vlayout = QVBoxLayout()
    right_side_vlayout.setAlignment(Qt.AlignTop)

    checkbox_layout, save_json_checkbox, save_model_checkbox = setup_save_checkboxes(
        parent
    )
    right_side_vlayout.addLayout(checkbox_layout)

    main_layout.addLayout(tab_method_layout, 35)
    main_layout.addWidget(QVerticalLineBreakWidget(parent))
    main_layout.addLayout(right_side_vlayout, 50)

    return (
        main_layout,
        tab_method_layout,
        right_side_vlayout,
        description_scroll,
        description_label,
        save_json_checkbox,
        save_model_checkbox,
    )
