from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout

from aydin.gui._qt.custom_widgets.readmoreless_label import QReadMoreLessLabel
from aydin.gui._qt.job_runners.previewall_job_runner import PreviewAllJobRunner
from aydin.gui._qt.transforms_tab_widget import TransformsTabWidget


class ProcessingTab(QWidget):
    """
    Better denoising with pre- and post-processing filters.

    Denoising quality can often be improved by applying carefully crafted transformations. Here you can activate and
    deactivate each transformation and its parameters. Each transformation consists in a pre- and post- processing
    step. In some cases the post-processing undoes the effects of the pre-processing, in other cases there is no
    meaningful post-processing. For each transformation you can choose to turn on or off the post-processing.
    <moreless>
    <split>
    We recommend that you test the effect of each transformation individually with the help of the per-transformation
    'preview' button, as well as the combined effect of all selected transforms with the help of 'preview all'
    button, before starting the denoising. Observe that we have selected some by default. Also keep in mind that
    previews will only display the post-processed as well as the pre-processed image and will not show the
    intervening effect of denoising.
    """

    def __init__(self, parent):
        super(ProcessingTab, self).__init__(parent)
        self.parent = parent

        self.tab_layout = QVBoxLayout()

        self.explanation_layout = QHBoxLayout()
        # Explanation text
        self.explanation_text = QReadMoreLessLabel(self, self.__doc__)
        self.explanation_layout.addWidget(self.explanation_text, 90)
        self.explanation_layout.addStretch()

        self.previewall_job_runner = PreviewAllJobRunner(self, self.parent.threadpool)
        self.explanation_layout.addWidget(self.previewall_job_runner, 10)

        self.tab_layout.addLayout(self.explanation_layout)

        self.panes_widget = TransformsTabWidget(self)

        self.tab_layout.addWidget(self.panes_widget)

        self.setLayout(self.tab_layout)

        self.set_advanced_enabled(enable=False)  # to init the tab correctly

    @property
    def transforms(self):
        if len(self.panes_widget.list_of_item_widgets) < 1:
            return None

        transforms = []

        for item in self.panes_widget.list_of_item_widgets:
            if item.params_dict:
                transforms.append(item.params_dict)

        return transforms

    def set_advanced_enabled(self, enable: bool = False):
        self.panes_widget.set_advanced_enabled(enable=enable)
