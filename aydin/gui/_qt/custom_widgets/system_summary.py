"""System resource summary widget showing CPU, memory, and GPU information."""

import os
import platform
import subprocess

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from aydin.util.misc.units import human_readable_byte_size


def _color_row(color, *labels):
    """Apply a color stylesheet to one or more QLabel widgets."""
    style = f"QLabel {{color: {color};}}"
    for label in labels:
        label.setStyleSheet(style)


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

        self._build_cpu_section()
        self._build_memory_section()
        self._build_gpu_section()

        self.main_layout.addWidget(self.cpu_group_box)
        self.main_layout.addWidget(self.memory_group_box)
        self.main_layout.addWidget(self.gpu_group_box)

        self.setLayout(self.main_layout)

    def _build_cpu_section(self):
        """Build the CPU summary group box with frequency, cores, and load."""
        self.cpu_group_box = QGroupBox("CPU Summary")
        grid = QGridLayout()
        grid.setSpacing(0)
        grid.setAlignment(Qt.AlignTop)
        self.cpu_group_box.setLayout(grid)

        row = 0

        # CPU freq
        freq_text = self._get_cpu_freq()
        cpu_freq_name = QLabel("Current CPU frequency:", self)
        self.cpu_freq_stats_label = QLabel(freq_text, self)
        grid.addWidget(cpu_freq_name, row, 0)
        grid.addWidget(self.cpu_freq_stats_label, row, 1)
        row += 1

        # Number of cores
        physical_cores = self._get_physical_cores()
        nb_cores_name = QLabel("Number of CPU cores:", self)
        self.nb_cores_label = QLabel(str(physical_cores), self)
        grid.addWidget(nb_cores_name, row, 0)
        grid.addWidget(self.nb_cores_label, row, 1)
        row += 1
        if isinstance(physical_cores, int):
            if physical_cores < 4:
                _color_row("red", nb_cores_name, self.nb_cores_label)
            elif physical_cores <= 6:
                _color_row("orange", nb_cores_name, self.nb_cores_label)
            else:
                _color_row("green", nb_cores_name, self.nb_cores_label)

        self.cpu_load_values = self._get_cpu_load()

        load_0 = self.cpu_load_values[0]
        load_1 = self.cpu_load_values[1]
        load_2 = self.cpu_load_values[2]
        val_0 = "100.0+%" if load_0 >= 100.0 else f"{round(load_0, 2)}%"
        val_1 = "100.0+%" if load_1 >= 100.0 else f"{round(load_1, 2)}%"
        val_2 = "100.0+%" if load_2 >= 100.0 else f"{round(load_2, 2)}%"

        load_name_0 = QLabel("CPU load over last 1min:", self)
        self.cpu_load_label0 = QLabel(val_0, self)
        grid.addWidget(load_name_0, row, 0)
        grid.addWidget(self.cpu_load_label0, row, 1)
        row += 1

        load_name_1 = QLabel("CPU load over last 5mins:", self)
        self.cpu_load_label1 = QLabel(val_1, self)
        grid.addWidget(load_name_1, row, 0)
        grid.addWidget(self.cpu_load_label1, row, 1)
        row += 1

        load_name_2 = QLabel("CPU load over last 15mins:", self)
        self.cpu_load_label2 = QLabel(val_2, self)
        grid.addWidget(load_name_2, row, 0)
        grid.addWidget(self.cpu_load_label2, row, 1)
        row += 1

        if self.cpu_load_values[0] >= 30:
            _color_row("red", load_name_0, self.cpu_load_label0)
        elif self.cpu_load_values[0] > 15:
            _color_row("orange", load_name_0, self.cpu_load_label0)
        else:
            _color_row("green", load_name_0, self.cpu_load_label0)
        if self.cpu_load_values[1] >= 30:
            _color_row("red", load_name_1, self.cpu_load_label1)
        elif self.cpu_load_values[1] > 15:
            _color_row("orange", load_name_1, self.cpu_load_label1)
        else:
            _color_row("green", load_name_1, self.cpu_load_label1)
        if self.cpu_load_values[2] >= 30:
            _color_row("red", load_name_2, self.cpu_load_label2)
        elif self.cpu_load_values[2] > 15:
            _color_row("orange", load_name_2, self.cpu_load_label2)
        else:
            _color_row("green", load_name_2, self.cpu_load_label2)

    def _build_memory_section(self):
        """Build the Memory summary group box with free/total RAM."""
        self.memory_group_box = QGroupBox("Memory Summary")
        grid = QGridLayout()
        grid.setSpacing(0)
        grid.setAlignment(Qt.AlignTop)
        self.memory_group_box.setLayout(grid)

        mem_available, mem_total = self._get_memory_info()

        if mem_available is not None and mem_total is not None:
            free_name = QLabel("Free Memory:", self)
            self.free_memory_label = QLabel(
                f"{human_readable_byte_size(mem_available)}, "
                f"({round(100 * mem_available / mem_total, 2)}%)",
                self,
            )
            grid.addWidget(free_name, 0, 0)
            grid.addWidget(self.free_memory_label, 0, 1)
            if mem_available < 8000000000:
                _color_row("red", free_name, self.free_memory_label)
            elif mem_available < 32000000000:
                _color_row("orange", free_name, self.free_memory_label)
            else:
                _color_row("green", free_name, self.free_memory_label)

            total_name = QLabel("Total Memory:", self)
            self.total_memory_label = QLabel(human_readable_byte_size(mem_total), self)
            grid.addWidget(total_name, 1, 0)
            grid.addWidget(self.total_memory_label, 1, 1)
            if mem_total < 8000000000:
                _color_row("red", total_name, self.total_memory_label)
            elif mem_total < 32000000000:
                _color_row("orange", total_name, self.total_memory_label)
            else:
                _color_row("green", total_name, self.total_memory_label)
        else:
            free_name = QLabel("Free Memory:", self)
            self.free_memory_label = QLabel(
                "N/A (install psutil for memory info)", self
            )
            _color_row("orange", free_name, self.free_memory_label)
            grid.addWidget(free_name, 0, 0)
            grid.addWidget(self.free_memory_label, 0, 1)

            total_name = QLabel("Total Memory:", self)
            self.total_memory_label = QLabel(
                "N/A (install psutil for memory info)", self
            )
            _color_row("orange", total_name, self.total_memory_label)
            grid.addWidget(total_name, 1, 0)
            grid.addWidget(self.total_memory_label, 1, 1)

    def _build_gpu_section(self):
        """Build the GPU summary group box with CUDA or MPS info."""
        self.gpu_group_box = QGroupBox("GPU Summary")
        grid = QGridLayout()
        grid.setSpacing(0)
        grid.setAlignment(Qt.AlignTop)
        self.gpu_group_box.setLayout(grid)

        cuda_gpu_name = self._get_cuda_gpu_name()
        mps_available = self._get_mps_available()

        if cuda_gpu_name != "N/A":
            # CUDA GPU path
            gpu_name = QLabel("GPU:", self)
            self.gpu_label = QLabel(cuda_gpu_name, self)
            _color_row("green", gpu_name, self.gpu_label)
            grid.addWidget(gpu_name, 0, 0)
            grid.addWidget(self.gpu_label, 0, 1)

            cuda_toolkit = self._get_cuda_toolkit_available()
            toolkit_name = QLabel("Toolkit:", self)
            self.toolkit_label = QLabel(
                "CUDA present" if cuda_toolkit else "CUDA absent", self
            )
            color = "green" if cuda_toolkit else "red"
            _color_row(color, toolkit_name, self.toolkit_label)
            grid.addWidget(toolkit_name, 1, 0)
            grid.addWidget(self.toolkit_label, 1, 1)

            mem_free, mem_total = self._get_cuda_memory()

            free_name = QLabel("Free Memory:", self)
            self.gpu_memory_free_label = QLabel(
                human_readable_byte_size(mem_free), self
            )
            grid.addWidget(free_name, 2, 0)
            grid.addWidget(self.gpu_memory_free_label, 2, 1)

            total_name = QLabel("Total Memory:", self)
            self.gpu_memory_total_label = QLabel(
                human_readable_byte_size(mem_total), self
            )
            grid.addWidget(total_name, 3, 0)
            grid.addWidget(self.gpu_memory_total_label, 3, 1)

            if mem_total == 0:
                _color_row("red", total_name, self.gpu_memory_total_label)
                _color_row("red", free_name, self.gpu_memory_free_label)
            else:
                if mem_total < 8000000000:
                    _color_row("orange", total_name, self.gpu_memory_total_label)
                else:
                    _color_row("green", total_name, self.gpu_memory_total_label)

                if mem_free / mem_total < 0.4:
                    _color_row("red", free_name, self.gpu_memory_free_label)
                elif mem_free / mem_total < 0.8:
                    _color_row("orange", free_name, self.gpu_memory_free_label)
                else:
                    _color_row("green", free_name, self.gpu_memory_free_label)

        elif mps_available:
            # Apple Silicon MPS path
            gpu_name = QLabel("GPU:", self)
            self.gpu_label = QLabel("Apple Silicon GPU (MPS)", self)
            _color_row("green", gpu_name, self.gpu_label)
            grid.addWidget(gpu_name, 0, 0)
            grid.addWidget(self.gpu_label, 0, 1)

            toolkit_name = QLabel("Toolkit:", self)
            self.toolkit_label = QLabel("MPS", self)
            _color_row("green", toolkit_name, self.toolkit_label)
            grid.addWidget(toolkit_name, 1, 0)
            grid.addWidget(self.toolkit_label, 1, 1)

            # Apple Silicon uses unified memory — show system memory info
            mem_available, mem_total = self._get_memory_info()
            if mem_available is not None and mem_total is not None:
                free_name = QLabel("Free Memory:", self)
                self.gpu_memory_free_label = QLabel(
                    human_readable_byte_size(mem_available), self
                )
                _color_row("green", free_name, self.gpu_memory_free_label)
                grid.addWidget(free_name, 2, 0)
                grid.addWidget(self.gpu_memory_free_label, 2, 1)

                total_name = QLabel("Total Memory:", self)
                self.gpu_memory_total_label = QLabel(
                    human_readable_byte_size(mem_total), self
                )
                _color_row("green", total_name, self.gpu_memory_total_label)
                grid.addWidget(total_name, 3, 0)
                grid.addWidget(self.gpu_memory_total_label, 3, 1)
            else:
                free_name = QLabel("Free Memory:", self)
                self.gpu_memory_free_label = QLabel("N/A (install psutil)", self)
                _color_row("orange", free_name, self.gpu_memory_free_label)
                grid.addWidget(free_name, 2, 0)
                grid.addWidget(self.gpu_memory_free_label, 2, 1)

                total_name = QLabel("Total Memory:", self)
                self.gpu_memory_total_label = QLabel("N/A (install psutil)", self)
                _color_row("orange", total_name, self.gpu_memory_total_label)
                grid.addWidget(total_name, 3, 0)
                grid.addWidget(self.gpu_memory_total_label, 3, 1)

        else:
            # No GPU available
            gpu_name = QLabel("GPU:", self)
            self.gpu_label = QLabel("N/A", self)
            _color_row("red", gpu_name, self.gpu_label)
            grid.addWidget(gpu_name, 0, 0)
            grid.addWidget(self.gpu_label, 0, 1)

            toolkit_name = QLabel("Toolkit:", self)
            self.toolkit_label = QLabel("N/A", self)
            _color_row("red", toolkit_name, self.toolkit_label)
            grid.addWidget(toolkit_name, 1, 0)
            grid.addWidget(self.toolkit_label, 1, 1)

            free_name = QLabel("Free Memory:", self)
            self.gpu_memory_free_label = QLabel(human_readable_byte_size(0), self)
            _color_row("red", free_name, self.gpu_memory_free_label)
            grid.addWidget(free_name, 2, 0)
            grid.addWidget(self.gpu_memory_free_label, 2, 1)

            total_name = QLabel("Total Memory:", self)
            self.gpu_memory_total_label = QLabel(human_readable_byte_size(0), self)
            _color_row("red", total_name, self.gpu_memory_total_label)
            grid.addWidget(total_name, 3, 0)
            grid.addWidget(self.gpu_memory_total_label, 3, 1)

    @staticmethod
    def _get_cpu_freq():
        """Return current CPU frequency as a formatted string.

        On ARM Macs where ``psutil.cpu_freq()`` returns ``None``, falls
        back to reading the chip name via ``sysctl`` or
        ``platform.processor()``.
        """
        try:
            import psutil

            cpu_freq = psutil.cpu_freq()
            if cpu_freq is not None:
                return f"{round(cpu_freq.current, 2)} Mhz"
        except ImportError:
            pass

        # Fallback for ARM Macs (and other platforms where cpu_freq is None)
        if platform.system() == "Darwin":
            try:
                brand = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
                if brand:
                    return brand
            except (subprocess.SubprocessError, OSError):
                pass

        proc = platform.processor()
        if proc:
            return proc
        return "N/A"

    @staticmethod
    def _get_physical_cores():
        """Return the number of physical CPU cores."""
        try:
            import psutil

            return psutil.cpu_count(logical=False) or os.cpu_count() or 1
        except ImportError:
            return os.cpu_count() or 1

    @staticmethod
    def _get_cpu_load():
        """Return CPU load averages as percentage list [1min, 5min, 15min]."""
        cpu_count = os.cpu_count() or 1
        try:
            import psutil

            return [(elem / cpu_count) * 100 for elem in psutil.getloadavg()]
        except ImportError:
            try:
                return [(elem / cpu_count) * 100 for elem in os.getloadavg()]
            except (OSError, AttributeError):
                return [0.0, 0.0, 0.0]

    @staticmethod
    def _get_memory_info():
        """Return (available, total) memory in bytes, or (None, None) if unavailable."""
        try:
            import psutil

            vm = psutil.virtual_memory()
            return vm.available, vm.total
        except ImportError:
            return None, None

    @staticmethod
    def _get_mps_available():
        """Return whether Apple Silicon MPS (Metal) GPU acceleration is available."""
        try:
            import torch

            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except ImportError:
            return False

    @staticmethod
    def _get_cuda_gpu_name():
        """Return the CUDA GPU device name, or 'N/A' if unavailable."""
        try:
            import numba
            from numba.cuda import CudaSupportError

            try:
                return numba.cuda.get_current_device().name.decode()
            except CudaSupportError:
                return "N/A"
        except ImportError:
            return "N/A"

    @staticmethod
    def _get_cuda_toolkit_available():
        """Return whether the CUDA toolkit is available."""
        try:
            import numba
            from numba.cuda import CudaSupportError

            try:
                return numba.cuda.cudadrv.nvvm.is_available()
            except CudaSupportError:
                return False
        except ImportError:
            return False

    @staticmethod
    def _get_cuda_memory():
        """Return (free, total) CUDA GPU memory in bytes."""
        try:
            import numba
            from numba.cuda import CudaSupportError

            try:
                mem_info = numba.cuda.current_context().get_memory_info()
                return mem_info.free, mem_info.total
            except CudaSupportError:
                return 0, 0
        except ImportError:
            return 0, 0
