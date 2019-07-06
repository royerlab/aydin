from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QProgressBar, QPlainTextEdit, \
    QSplitter

from pitl.gui.components.filepath_picker import FilePathPicker


class Noise2TruthTab(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

        self.layout = QVBoxLayout()

        """
        Paths layout where we list required paths and what are current values for those
        Also, these boxes are drag-and-drop areas. User drag-and-drop any file or folder,
        or user can set the path with the help of button on the right end.
        """
        paths_layout = QVBoxLayout()
        paths_layout.addWidget(QLabel("Path for the input training noisy images: "))
        paths_layout.addWidget(FilePathPicker())
        paths_layout.addWidget(QLabel("Path for the input training groundtruth images: "))
        paths_layout.addWidget(FilePathPicker())
        paths_layout.addWidget(QLabel("Path for the input test noisy images: "))
        paths_layout.addWidget(FilePathPicker())
        paths_layout.addWidget(QLabel("Path for the resulting output images: "))
        paths_layout.addWidget(FilePathPicker())

        # Buttons layout where we have run button and other functional methods
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(QPushButton("Train"))
        buttons_layout.addWidget(QPushButton("Test"))


        # Build splitter
        def_splitter = QSplitter(Qt.Vertical)

        def_splitter.addWidget(QPlainTextEdit())  # for tab definitions

        paths_and_buttons_layout = QVBoxLayout()
        paths_and_buttons_layout.addLayout(paths_layout)
        paths_and_buttons_layout.addLayout(buttons_layout)
        paths_and_buttons_layout.addWidget(QProgressBar(self))

        paths_and_buttons = QWidget()
        paths_and_buttons.setLayout(paths_and_buttons_layout)
        def_splitter.addWidget(paths_and_buttons)

        # Add splitter into main layout
        self.layout.addWidget(def_splitter)
        self.setLayout(self.layout)
