from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel


class AboutTab(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

        self.layout = QVBoxLayout()

        self.layout.addWidget(QLabel("Ahmet Can Solak"))
        self.layout.addWidget(QLabel("Hirofumi Kobayashi"))
        self.layout.addWidget(QLabel("Josh Batson"))
        self.layout.addWidget(QLabel("Loic Royer"))

        self.setLayout(self.layout)
