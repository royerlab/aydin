from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QListWidget,
)

from aydin.gui._qt.custom_widgets.denoise_tab_method import DenoiseTabMethodWidget
from aydin.gui._qt.custom_widgets.horizontal_line_break_widget import (
    QHorizontalLineBreakWidget,
)
from aydin.gui._qt.custom_widgets.readmoreless_label import QReadMoreLessLabel
from aydin.restoration.denoise.util.denoise_utils import (
    get_list_of_denoiser_implementations,
)


class DenoiseTab(QWidget):
    """
    Now it is time to denoise the previously selected and cropped images.
    <br><br>
    Aydin comes with a growing variety of self-supervised, auto-tuned, and unsupervised denoising algorithms,
    each with their own strengths and weaknesses in terms of speed, denoising performance, artifacts, and propensity
    to hallucinate unsubstantiated details. We recommend you check our
    <a href='https://royerlab.github.io/aydin/use_cases/introduction.html'>use cases</a> to learn how to
    choose the best algorithm and parameters for your image(s).

    <moreless>
    Quick guide: The first algorithm to try is 'Butterworth' which is remarkably simple, fast, and sometimes
    embarrassingly effective compared to more sophisticated methods. Next you can try the Gaussian-Median mixed
    denoiser (gm) which is in some cases quite effective. 'Spectral' can give extremely good results too but can take
    longer, as do in general patch and dictionary based methods.
    <split>
    Our own favourite is a novel variant on the Noise2Self theme which relies on carefully crafted features and
    gradient boosting (N2S-FGR-cb or -lgbm). CNN-based Noise2Self denoising is also available but is currently on of
    our least favourites because of its propensity to hallucinate detail, slowness, and overall worse performance. In
    fine, there is no silver bullet, there is not a single denoising algorithm that can tackle all denoising
    challenges, instead you need to choose and play with a variety of algorithms to find the one that will fit you
    needs both in terms of processing speed, visual appearance, and downstream analysis constraints. Denoising is a
    form of image analysis that consists in separating signal from noise, with the definition of signal and noise
    being to some extent, subjective and context dependent.
    """

    def __init__(self, parent):
        super(DenoiseTab, self).__init__(parent)
        self.parent = parent

        self.tab_layout = QVBoxLayout()
        self.tab_layout.setAlignment(Qt.AlignTop)

        self.tab_layout.addWidget(QReadMoreLessLabel(self, self.__doc__))

        # Horizontal Line Break
        self.tab_layout.addWidget(QHorizontalLineBreakWidget(self))

        self.leftlist = QListWidget()

        (
            backend_options,
            backend_options_descriptions,
        ) = get_list_of_denoiser_implementations()

        self.backend_options, self.backend_options_descriptions = (
            list(t)
            for t in zip(*sorted(zip(backend_options, backend_options_descriptions)))
        )

        self.basic_backend_options = [
            'Classic-butterworth',
            'Classic-gaussian',
            'Classic-gm',
            'Classic-nlm',
            'Classic-tv',
            'Noise2SelfCNN-jinet',
            'Noise2SelfFGR-cb',
            'Noise2SelfFGR-lgbm',
            'Noise2SelfFGR-random_forest',
        ]

        self.basic_backend_options_descriptions = []

        self.stacked_widget = QStackedWidget(self)

        default_option_index = 0
        for idx, (backend_option, description) in enumerate(
            zip(self.backend_options, self.backend_options_descriptions)
        ):
            if backend_option in self.basic_backend_options:
                self.basic_backend_options_descriptions.append(description)

                self.leftlist.insertItem(idx, backend_option)

                self.stacked_widget.addWidget(
                    DenoiseTabMethodWidget(
                        self, name=backend_option, description=description
                    )
                )

                if backend_option == "Classic-butterworth":
                    default_option_index = idx

        self.leftlist.item(default_option_index).setSelected(True)
        self.change_current_method(default_option_index)
        self.leftlist.currentRowChanged.connect(self.change_current_method)

        hbox = QHBoxLayout()
        hbox.addWidget(self.leftlist, 15)
        hbox.addWidget(self.stacked_widget, 85)

        self.tab_layout.addLayout(hbox)

        self.setLayout(self.tab_layout)

        self.set_advanced_enabled(enable=False)  # to init the tab correctly

    def change_current_method(self, new_index):
        self.stacked_widget.setCurrentIndex(new_index)

    @property
    def selected_backend(self):
        return self.stacked_widget.currentWidget().name

    @property
    def current_backend_widget(self):
        return self.stacked_widget.currentWidget()

    @property
    def lower_level_args(self):
        return self.stacked_widget.currentWidget().lower_level_args()

    def set_advanced_enabled(self, enable: bool = False):
        self.leftlist.clear()

        while self.stacked_widget.count():
            self.stacked_widget.removeWidget(self.stacked_widget.widget(0))

        if enable:
            options = self.backend_options
            description_list = self.backend_options_descriptions
        else:
            options = self.basic_backend_options
            description_list = self.basic_backend_options_descriptions

        for index, backend_option in enumerate(options):
            self.leftlist.insertItem(index, backend_option)

            self.stacked_widget.addWidget(
                DenoiseTabMethodWidget(
                    self, name=backend_option, description=description_list[index]
                )
            )

        for widget_index in range(self.stacked_widget.count()):
            for key, constructor_arguments_widget in self.stacked_widget.widget(
                widget_index
            ).constructor_arguments_widget_dict.items():
                constructor_arguments_widget.set_advanced_enabled(enable=enable)
