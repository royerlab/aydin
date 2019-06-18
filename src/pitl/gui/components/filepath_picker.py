from PyQt5.Qt import *
import sys


class FilePathPicker(QWidget):
    """
    Subclass the widget and add a button to load images.
    Alternatively set up dragging and dropping of image files onto the widget
    """

    def __init__(self):
        super(FilePathPicker, self).__init__()
        self.filename = None

        # Button that allows loading of images
        self.load_button = QPushButton("Load file path")
        self.load_button.clicked.connect(self.load_file_button)

        # Path viewing region
        self.lbl = QLabel(self)

        # A horizontal layout to include the button on the left
        layout_button = QHBoxLayout()
        layout_button.addWidget(self.load_button)
        layout_button.addStretch()

        # A Vertical layout to include the button layout and then the image
        layout = QHBoxLayout()
        layout.addLayout(layout_button)
        layout.addWidget(self.lbl)

        self.setLayout(layout)

        # Enable dragging and dropping onto the GUI
        self.setAcceptDrops(True)

        self.show()

    def load_file_button(self):
        """
        Open a File dialog when the button is pressed

        :return:
        """

        # Get the file location

        self.filename, _ = QFileDialog.getOpenFileName(QFileDialog(), 'Open file')
        # Load the image from the location
        self.load_file()

    def load_file(self):
        """
        Set the fname to label

        :return:
        """
        self.lbl.setText(self.filename)

    # The following three methods set up dragging and dropping for the app
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        """
        Drop files directly onto the widget
        File locations are stored in fname

        :param e:
        :return:
        """
        if e.mimeData().hasUrls:
            e.setDropAction(Qt.CopyAction)
            e.accept()

            for url in e.mimeData().urls():
                    fname = str(url.toLocalFile())

            self.filename = fname
            self.load_file()
        else:
            e.ignore()


# Demo, Runs if called directly
if __name__ == '__main__':
    # Initialise the application
    app = QApplication(sys.argv)
    # Call the widget
    ex = FilePathPicker()
    sys.exit(app.exec_())
