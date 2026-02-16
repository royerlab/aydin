"""System resource summary widget showing CPU, memory, and GPU information."""

import os

import numba
import psutil
from numba.cuda import CudaSupportError
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from aydin.util.misc.units import human_readable_byte_size


class SystemSummaryWidget(QWidget):
    """Widget displaying a summary of the system's CPU, memory, and GPU resources.

    Shows CPU frequency, core count, load averages, free/total RAM, CUDA GPU
    name, toolkit availability, and GPU memory. Values are color-coded
    (green/orange/red) to indicate resource adequacy.

    Parameters
    ----------
    parent : QWidget
        The parent widget.
    """

    def __init__(self, parent):
        """Initialize the system summary by querying CPU, memory, and GPU info.

        Parameters
        ----------
        parent : QWidget
            The parent widget.
        """
        QWidget.__init__(self, parent)

        self.main_layout = QHBoxLayout()
        self.main_layout.setAlignment(Qt.AlignCenter)

        # CPU summary
        self.cpu_group_box = QGroupBox("CPU Summary")
        self.cpu_group_box_layout = QVBoxLayout()
        self.cpu_group_box_layout.setSpacing(0)
        self.cpu_group_box_layout.setAlignment(Qt.AlignTop)
        self.cpu_group_box.setLayout(self.cpu_group_box_layout)

        # CPU freq
        cpu_freq = psutil.cpu_freq()
        freq_text = f"{round(cpu_freq.current, 2)} Mhz" if cpu_freq else "N/A"
        self.cpu_freq_stats_label = QLabel(
            f"Current CPU frequency:\t {freq_text}", self
        )
        self.cpu_group_box_layout.addWidget(self.cpu_freq_stats_label)

        # Number of cores
        physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        self.nb_cores_label = QLabel(f"Number of CPU cores:\t {physical_cores}", self)
        self.cpu_group_box_layout.addWidget(self.nb_cores_label)
        if physical_cores < 4:
            self.nb_cores_label.setStyleSheet("QLabel {color: red;}")
        elif physical_cores <= 6:
            self.nb_cores_label.setStyleSheet("QLabel {color: orange;}")
        else:
            self.nb_cores_label.setStyleSheet("QLabel {color: green;}")

        cpu_count = os.cpu_count() or 1
        self.cpu_load_values = [
            (elem / cpu_count) * 100 for elem in psutil.getloadavg()
        ]

        self.cpu_load_label0 = QLabel(
            f"CPU load over last 1min:\t {'100.0+' if self.cpu_load_values[0] >= 100.0 else round(self.cpu_load_values[0], 2)}%",
            self,
        )
        self.cpu_group_box_layout.addWidget(self.cpu_load_label0)

        self.cpu_load_label1 = QLabel(
            f"CPU load over last 5mins:\t {'100.0+' if self.cpu_load_values[1] >= 100.0 else round(self.cpu_load_values[1], 2)}%",
            self,
        )
        self.cpu_group_box_layout.addWidget(self.cpu_load_label1)

        self.cpu_load_label2 = QLabel(
            f"CPU load over last 15mins:\t {'100.0+' if self.cpu_load_values[2] >= 100.0 else round(self.cpu_load_values[2], 2)}%",
            self,
        )
        self.cpu_group_box_layout.addWidget(self.cpu_load_label2)

        if self.cpu_load_values[0] >= 30:
            self.cpu_load_label0.setStyleSheet("QLabel {color: red;}")
        elif self.cpu_load_values[0] > 15:
            self.cpu_load_label0.setStyleSheet("QLabel {color: orange;}")
        else:
            self.cpu_load_label0.setStyleSheet("QLabel {color: green;}")
        if self.cpu_load_values[1] >= 30:
            self.cpu_load_label1.setStyleSheet("QLabel {color: red;}")
        elif self.cpu_load_values[1] > 15:
            self.cpu_load_label1.setStyleSheet("QLabel {color: orange;}")
        else:
            self.cpu_load_label1.setStyleSheet("QLabel {color: green;}")
        if self.cpu_load_values[2] >= 30:
            self.cpu_load_label2.setStyleSheet("QLabel {color: red;}")
        elif self.cpu_load_values[2] > 15:
            self.cpu_load_label2.setStyleSheet("QLabel {color: orange;}")
        else:
            self.cpu_load_label2.setStyleSheet("QLabel {color: green;}")

        # Memory summary
        self.memory_group_box = QGroupBox("Memory Summary")
        self.memory_group_box_layout = QVBoxLayout()
        self.memory_group_box_layout.setSpacing(0)
        self.memory_group_box_layout.setAlignment(Qt.AlignTop)
        self.memory_group_box.setLayout(self.memory_group_box_layout)

        self.free_memory_label = QLabel(
            f"Free Memory:\t {human_readable_byte_size(psutil.virtual_memory().available)}, "
            f"({round(100 * psutil.virtual_memory().available / psutil.virtual_memory().total, 2)}%)",
            self,
        )
        self.memory_group_box_layout.addWidget(self.free_memory_label)
        if psutil.virtual_memory().available < 8000000000:
            self.free_memory_label.setStyleSheet("QLabel {color: red;}")
        elif psutil.virtual_memory().available < 32000000000:
            self.free_memory_label.setStyleSheet("QLabel {color: orange;}")
        else:
            self.free_memory_label.setStyleSheet("QLabel {color: green;}")

        self.total_memory_label = QLabel(
            f"Total Memory:\t {human_readable_byte_size(psutil.virtual_memory().total)}",
            self,
        )
        self.memory_group_box_layout.addWidget(self.total_memory_label)
        if psutil.virtual_memory().total < 8000000000:
            self.total_memory_label.setStyleSheet("QLabel {color: red;}")
        elif psutil.virtual_memory().total < 32000000000:
            self.total_memory_label.setStyleSheet("QLabel {color: orange;}")
        else:
            self.total_memory_label.setStyleSheet("QLabel {color: green;}")

        # GPU summary
        self.gpu_group_box = QGroupBox("GPU Summary")
        self.gpu_group_box_layout = QVBoxLayout()
        self.gpu_group_box_layout.setSpacing(0)
        self.gpu_group_box_layout.setAlignment(Qt.AlignTop)
        self.gpu_group_box.setLayout(self.gpu_group_box_layout)

        try:
            cuda_gpu_name = numba.cuda.get_current_device().name.decode()
        except CudaSupportError:
            cuda_gpu_name = "N/A"

        self.cuda_gpu_label = QLabel(f"CUDA GPU: \t\t{cuda_gpu_name}", self)
        self.gpu_group_box_layout.addWidget(self.cuda_gpu_label)

        if cuda_gpu_name != "N/A":
            self.cuda_gpu_label.setStyleSheet("QLabel {color: green;}")
        else:
            self.cuda_gpu_label.setStyleSheet("QLabel {color: red;}")

        cuda_toolkit = numba.cuda.cudadrv.nvvm.is_available()
        self.cudatoolkit_label = QLabel(
            f"CUDA Toolkit: \t\t{'present' if cuda_toolkit else 'absent'}", self
        )
        self.gpu_group_box_layout.addWidget(self.cudatoolkit_label)

        if numba.cuda.cudadrv.nvvm.is_available():
            self.cudatoolkit_label.setStyleSheet("QLabel {color: green;}")
        else:
            self.cudatoolkit_label.setStyleSheet("QLabel {color: red;}")

        try:
            cuda_memory_free = numba.cuda.current_context().get_memory_info().free
            cuda_memory_total = numba.cuda.current_context().get_memory_info().total
        except CudaSupportError:
            cuda_memory_free = 0
            cuda_memory_total = 0

        self.gpu_memory_free_label = QLabel(
            f"Free GPU Memory: \t{human_readable_byte_size(cuda_memory_free)}", self
        )
        self.gpu_group_box_layout.addWidget(self.gpu_memory_free_label)

        self.gpu_memory_total_label = QLabel(
            f"Total GPU Memory: \t{human_readable_byte_size(cuda_memory_total)}", self
        )
        self.gpu_group_box_layout.addWidget(self.gpu_memory_total_label)

        if cuda_memory_total == 0:
            self.gpu_memory_total_label.setStyleSheet("QLabel {color: red;}")
            self.gpu_memory_free_label.setStyleSheet("QLabel {color: red;}")
        else:
            if numba.cuda.current_context().get_memory_info().total < 8000000000:
                self.gpu_memory_total_label.setStyleSheet("QLabel {color: orange;}")
            else:
                self.gpu_memory_total_label.setStyleSheet("QLabel {color: green;}")

            if cuda_memory_free / cuda_memory_total < 0.4:
                self.gpu_memory_free_label.setStyleSheet("QLabel {color: red;}")
            elif cuda_memory_free / cuda_memory_total < 0.8:
                self.gpu_memory_free_label.setStyleSheet("QLabel {color: orange;}")
            else:
                self.gpu_memory_free_label.setStyleSheet("QLabel {color: green;}")

        self.main_layout.addWidget(self.cpu_group_box)
        self.main_layout.addWidget(self.memory_group_box)
        self.main_layout.addWidget(self.gpu_group_box)

        self.setLayout(self.main_layout)
