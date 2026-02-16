"""Custom range slider widgets with two draggable handles."""

from qtpy.QtCore import Property, Qt, Signal
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import QWidget


class QRangeSlider(QWidget):
    """Base class for range sliders with two draggable handles.

    Provides the core logic for a dual-handle slider that selects a min/max
    range. Handles mouse interaction, value/display conversion, collapse/expand
    behavior, and property-based color customization. Subclass ``QHRangeSlider``
    for the horizontal variant.

    Signals
    -------
    valuesChanged : Signal(tuple)
        Emitted when the slider values change.
    rangeChanged : Signal(tuple)
        Emitted when the slider range changes.
    collapsedChanged : Signal(bool)
        Emitted when the slider collapse state changes.
    focused : Signal
        Emitted when the slider receives a mouse press.
    resized : Signal
        Emitted when the slider widget is resized.
    """

    valuesChanged = Signal(tuple)
    rangeChanged = Signal(tuple)
    collapsedChanged = Signal(bool)
    focused = Signal()
    resized = Signal()

    def __init__(
        self,
        initial_values=None,
        data_range=None,
        step_size=None,
        collapsible=True,
        collapsed=False,
        parent=None,
    ):
        """A range slider with two handles for min/max values.

        Values should be provided in the range of the underlying data.
        (normalization to 0-1 happens internally in the slider.slider_values())

        Parameters
        ----------
        initial_values : 2-tuple, optional
            Initial min & max values of the slider, defaults to (0.2, 0.8)
        data_range : 2-tuple, optional
            Min and max of the slider range, defaults to (0, 1)
        step_size : float, optional
            Single step size for the slider, defaults to 1
        collapsible : bool
            Whether the slider is collapsible, defaults to True.
        collapsed : bool
            Whether the slider begins collapsed, defaults to False.
        parent : qtpy.QtWidgets.QWidget
            Parent widget.
        """
        super(QRangeSlider, self).__init__(parent)
        self.handle_radius = 8
        self.slider_width = 6
        self.moving = "none"
        self.collapsible = collapsible
        self.collapsed = collapsed
        self.prev_moving = None
        self.bc_min = None
        self.bc_max = None

        # Variables initialized in methods
        self.value_min = 0
        self.value_max = 1
        self.start_display_min = None
        self.start_display_max = None
        self.start_pos = None
        self.display_min = None
        self.display_max = None

        self.setBarColor(QColor(200, 200, 200))
        self.setBackgroundColor(QColor(100, 100, 100))
        self.setHandleColor(QColor(200, 200, 200))
        self.setHandleBorderColor(QColor(200, 200, 200))

        self.set_range((0, 100) if data_range is None else data_range)
        self.set_values((20, 80) if initial_values is None else initial_values)
        if step_size is None:
            # pick an appropriate slider step size based on the data range
            if data_range is not None:
                step_size = (data_range[1] - data_range[0]) / 1000
            else:
                step_size = 0.001
        self.set_step(step_size)
        if not parent:
            if 'HRange' in self.__class__.__name__:
                self.setGeometry(200, 200, 200, 20)
            else:
                self.setGeometry(200, 200, 20, 200)

    def range(self):
        """Return the min and max possible values for the slider range.

        Returns
        -------
        tuple of float
            ``(min, max)`` range boundaries in data units.
        """
        return self.data_range_min, self.data_range_max

    def set_range(self, values):
        """Set the min and max possible values for the slider range.

        Parameters
        ----------
        values : 2-tuple of float
            ``(min, max)`` range boundaries in data units.
        """
        self.data_range_min, self.data_range_max = values
        self.rangeChanged.emit(self.range())
        self.update_display_positions()

    def values(self):
        """Current slider values.

        Returns
        -------
        tuple
            Current minimum and maximum values of the range slider
        """
        return tuple([self._slider_to_data_value(v) for v in self.slider_values()])

    def set_values(self, values):
        """Set the slider min/max values in data units.

        Parameters
        ----------
        values : 2-tuple of float
            New (min, max) values in data units.
        """
        self.set_slider_values([self._data_to_slider_value(v) for v in values])

    def slider_values(self):
        """Current slider values, as a fraction of slider width.

        Returns
        -------
        values : 2-tuple of int
            Start and end of the range.
        """
        return self.value_min, self.value_max

    def set_slider_values(self, values):
        """Set current slider values, as a fraction of slider width.

        Parameters
        ----------
        values : 2-tuple of float or int
            Start and end of the range.
        """
        self.value_min, self.value_max = values
        self.valuesChanged.emit(self.values())
        self.update_display_positions()

    def set_step(self, step):
        """Set the step size for slider movement.

        Parameters
        ----------
        step : float
            Single step size in data units.
        """
        self._step = step

    @property
    def single_step(self):
        """Step size normalized to slider scale (0-1 range).

        Returns
        -------
        float
            Normalized step size.
        """
        return self._step / self.scale

    def mouseMoveEvent(self, event):
        """Handle mouse drag to update the slider handle positions.

        Parameters
        ----------
        event : QMouseEvent
            The mouse move event.
        """
        if not self.isEnabled():
            return

        size = self.range_slider_size()
        pos = self.get_pos(event)
        if self.moving == "min":
            if pos <= self.handle_radius:
                self.display_min = self.handle_radius
            elif pos > self.display_max - self.handle_radius / 2:
                self.display_min = self.display_max - self.handle_radius / 2
            else:
                self.display_min = pos
        elif self.moving == "max":
            if pos >= size + self.handle_radius:
                self.display_max = size + self.handle_radius
            elif pos < self.display_min + self.handle_radius / 2:
                self.display_max = self.display_min + self.handle_radius / 2
            else:
                self.display_max = pos
        elif self.moving == "bar":
            width = self.start_display_max - self.start_display_min
            lower_part = self.start_pos - self.start_display_min
            upper_part = self.start_display_max - self.start_pos
            if pos + upper_part >= size + self.handle_radius:
                self.display_max = size + self.handle_radius
                self.display_min = self.display_max - width
            elif pos - lower_part <= self.handle_radius:
                self.display_min = self.handle_radius
                self.display_max = self.display_min + width
            else:
                self.display_min = pos - lower_part
                self.display_max = self.display_min + width

        self.update_values_from_display()

    def mousePressEvent(self, event):
        """Handle mouse press to start dragging a handle or the range bar.

        Left-click selects the nearest handle or bar. Right-click toggles
        collapse/expand if the slider is collapsible.

        Parameters
        ----------
        event : QMouseEvent
            The mouse press event.
        """
        if not self.isEnabled():
            return

        pos = self.get_pos(event)
        top = self.range_slider_size() + self.handle_radius
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.collapsed:
                if abs(self.display_min - pos) <= self.handle_radius:
                    self.moving = "min"
                elif abs(self.display_max - pos) <= self.handle_radius:
                    self.moving = "max"
                elif self.display_min < pos < self.display_max:
                    self.moving = "bar"
                elif self.display_max < pos < top:
                    self.display_max = pos
                    self.moving = "max"
                    self.update_values_from_display()
                elif self.display_min > pos > self.handle_radius:
                    self.display_min = pos
                    self.moving = "min"
                    self.update_values_from_display()
            else:
                self.moving = "bar"
                if self.handle_radius < pos < top:
                    self.display_max = pos
                    self.display_min = pos
        else:
            if self.collapsible:
                if self.collapsed:
                    self.expand()
                else:
                    self.collapse()
                self.collapsedChanged.emit(self.collapsed)

        self.start_display_min = self.display_min
        self.start_display_max = self.display_max
        self.start_pos = pos
        self.focused.emit()

    def mouseReleaseEvent(self, event):
        """Emit valuesChanged on mouse release and reset the moving state.

        Parameters
        ----------
        event : QMouseEvent
            The mouse release event.
        """
        if self.isEnabled():
            if not (self.moving == "none"):
                self.valuesChanged.emit(self.values())
            self.moving = "none"

    def collapse(self):
        """Collapse the slider range to a single midpoint value."""
        self.bc_min, self.bc_max = self.value_min, self.value_max
        midpoint = (self.value_max + self.value_min) / 2
        min_value = midpoint
        max_value = midpoint
        self.set_slider_values((min_value, max_value))
        self.collapsed = True

    def expand(self):
        """Expand the slider range back to its pre-collapse extent."""
        _mid = (self.bc_max - self.bc_min) / 2
        min_value = self.value_min - _mid
        max_value = self.value_min + _mid
        if min_value < 0:
            min_value = 0
            max_value = self.bc_max - self.bc_min
        elif max_value > 1:
            max_value = 1
            min_value = max_value - (self.bc_max - self.bc_min)
        self.set_slider_values((min_value, max_value))
        self.collapsed = False

    def resizeEvent(self, event):
        """Recalculate display positions when the widget is resized.

        Parameters
        ----------
        event : QResizeEvent
            The resize event.
        """
        self.update_display_positions()
        self.resized.emit()

    def update_display_positions(self):
        """Recalculate the pixel positions of the slider handles from values."""
        size = self.range_slider_size()
        range_min = int(size * self.value_min)
        range_max = int(size * self.value_max)
        self.display_min = range_min + self.handle_radius
        self.display_max = range_max + self.handle_radius
        self.update()

    def _data_to_slider_value(self, value):
        """Convert a data-space value to the normalized slider scale (0-1).

        Parameters
        ----------
        value : float
            Value in data units.

        Returns
        -------
        float
            Normalized slider value between 0 and 1.
        """
        rmin, rmax = self.range()
        return (value - rmin) / self.scale

    def _slider_to_data_value(self, value):
        """Convert a normalized slider value (0-1) back to data-space units.

        Parameters
        ----------
        value : float
            Normalized slider value between 0 and 1.

        Returns
        -------
        float
            Value in data units.
        """
        rmin, rmax = self.range()
        return rmin + value * self.scale

    @property
    def scale(self):
        """Total range span in data units.

        Returns
        -------
        float
            Difference between data range max and min.
        """
        return self.data_range_max - self.data_range_min

    def update_values_from_display(self):
        """Update the internal slider values from the current display positions."""
        size = self.range_slider_size()
        val_min, val_max = self.slider_values()
        if (self.moving == "min") or (self.moving == "bar"):
            scale_min = (self.display_min - self.handle_radius) / size
            ratio = round(scale_min / self.single_step)
            val_min = ratio * self.single_step
        if (self.moving == "max") or (self.moving == "bar"):
            scale_max = (self.display_max - self.handle_radius) / size
            ratio = round(scale_max / self.single_step)
            val_max = ratio * self.single_step
        self.set_slider_values((val_min, val_max))

    def getBarColor(self):
        """Return the fill color of the selected range bar.

        Returns
        -------
        QColor
            Current bar color.
        """
        return self.bar_color

    def setBarColor(self, barColor):
        """Set the fill color of the selected range bar.

        Parameters
        ----------
        barColor : QColor
            New bar color.
        """
        self.bar_color = barColor

    barColor = Property(QColor, getBarColor, setBarColor)

    def getBackgroundColor(self):
        """Return the background color of the slider track.

        Returns
        -------
        QColor
            Current background color.
        """
        return self.background_color

    def setBackgroundColor(self, backgroundColor):
        """Set the background color of the slider track.

        Parameters
        ----------
        backgroundColor : QColor
            New background color.
        """
        self.background_color = backgroundColor

    backgroundColor = Property(QColor, getBackgroundColor, setBackgroundColor)

    @property
    def handle_width(self):
        """Diameter of the slider handles in pixels.

        Returns
        -------
        int
            Handle width (2 * handle_radius).
        """
        return self.handle_radius * 2

    def getHandleColor(self):
        """Return the fill color of the slider handles.

        Returns
        -------
        QColor
            Current handle fill color.
        """
        return self.handle_color

    def setHandleColor(self, handleColor):
        """Set the fill color of the slider handles.

        Parameters
        ----------
        handleColor : QColor
            New handle fill color.
        """
        self.handle_color = handleColor

    handleColor = Property(QColor, getHandleColor, setHandleColor)

    def getHandleBorderColor(self):
        """Return the border color of the slider handles.

        Returns
        -------
        QColor
            Current handle border color.
        """
        return self.handle_border_color

    def setHandleBorderColor(self, handleBorderColor):
        """Set the border color of the slider handles.

        Parameters
        ----------
        handleBorderColor : QColor
            New handle border color.
        """
        self.handle_border_color = handleBorderColor

    handleBorderColor = Property(QColor, getHandleBorderColor, setHandleBorderColor)

    def setEnabled(self, enabled):
        """Enable or disable the slider and trigger a visual update.

        Parameters
        ----------
        enabled : bool
            Whether to enable the slider.
        """
        super().setEnabled(enabled)
        self.update()


class QHRangeSlider(QRangeSlider):
    """Horizontal range slider with two draggable handles.

    Extends ``QRangeSlider`` with horizontal painting and layout. The slider
    renders a background track, a colored range bar between the two handles,
    and circular handle indicators.

    Parameters
    ----------
    initial_values : 2-tuple of float, optional
        Initial (min, max) values of the slider. Default is (0.2, 0.8).
    data_range : 2-tuple of float, optional
        Min and max of the slider range. Default is (0, 1).
    step_size : float, optional
        Single step size for the slider. Default is 1.
    collapsible : bool, optional
        Whether the slider is collapsible. Default is True.
    collapsed : bool, optional
        Whether the slider begins collapsed. Default is False.
    parent : QWidget, optional
        Parent widget.
    """

    def get_pos(self, event):
        """Get event position.

        Parameters
        ----------
        event : qtpy.QEvent
            Event from the Qt context.

        Returns
        -------
        position : int
            Relative horizontal position of the event.
        """
        return event.position().x()

    def paintEvent(self, event):
        """Paint the background, range bar and splitters.

        Parameters
        ----------
        event : qtpy.QEvent
            Event from the Qt context.
        """
        painter, w, h = QPainter(self), self.width(), self.height()

        half_width = self.slider_width / 2 - 1
        halfdiff = int(h / 2 - half_width)

        # Background
        painter.setPen(self.background_color)
        painter.setBrush(self.background_color)
        painter.drawRoundedRect(0, halfdiff, w, self.slider_width, 2, 2)

        # Range Bar
        painter.setPen(self.bar_color)
        painter.setBrush(self.bar_color)
        if self.collapsed:
            painter.drawRect(0, halfdiff, self.display_max, self.slider_width)
        else:
            painter.drawRect(
                self.display_min,
                halfdiff,
                self.display_max - self.display_min,
                self.slider_width,
            )

        painter.setRenderHints(QPainter.RenderHint.Antialiasing)
        # Splitters
        painter.setPen(self.handle_border_color)
        painter.setBrush(self.handle_color)
        painter.drawEllipse(
            self.display_min - self.handle_radius,
            int(h / 2 - self.handle_radius + 1),
            self.handle_width - 1,
            self.handle_width - 1,
        )  # left
        painter.drawEllipse(
            self.display_max - self.handle_radius,
            int(h / 2 - self.handle_radius + 1),
            self.handle_width - 1,
            self.handle_width - 1,
        )  # right

    def range_slider_size(self):
        """Width of the slider, in pixels

        Returns
        -------
        size : int
            Slider bar length (horizontal sliders) or height (vertical
            sliders).
        """
        return float(self.width() - self.handle_width)
