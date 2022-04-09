from qtpy.QtWidgets import QWidget

from aydin.gui._qt.custom_widgets.activity_widget import ActivityWidget


def test_hello(qtbot):
    dummy_parent = QWidget()
    widget = ActivityWidget(dummy_parent)
    qtbot.addWidget(widget)

    widget.activity_print("testinghard")
    widget.activity_print("testingharder")

    print(widget.infoTextBox.toPlainText())

    assert widget.infoTextBox.toPlainText() == "testinghard" + "testingharder"
