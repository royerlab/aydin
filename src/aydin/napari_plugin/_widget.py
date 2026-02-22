"""Aydin Denoiser napari dock widget (Tier 1)."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from aydin.napari_plugin._axes_utils import detect_axes_from_napari_layer

# (display name, variant string, tooltip)
# None variant means Classic with auto-selection (variant=None)
_METHODS = [
    (
        'N2S-FGR CatBoost (recommended)',
        'Noise2SelfFGR-cb',
        'Self-supervised denoising with Feature Generation & Regression '
        'using CatBoost gradient boosting. Best overall quality for most '
        'images. Medium speed.',
    ),
    (
        'N2S-FGR LightGBM',
        'Noise2SelfFGR-lgbm',
        'Self-supervised FGR using LightGBM. Similar quality to CatBoost, '
        'sometimes faster on large images.',
    ),
    (
        'N2S-CNN UNet',
        'Noise2SelfCNN-unet',
        'Self-supervised deep learning denoiser using a UNet architecture. '
        'Best for images with complex noise patterns. Slower, benefits from GPU.',
    ),
    (
        'Butterworth (fast)',
        'Classic-butterworth',
        'Classical frequency-domain filter. Very fast, good for periodic '
        'noise and smooth structures.',
    ),
    (
        'Gaussian',
        'Classic-gaussian',
        'Classical Gaussian smoothing filter. Very fast and simple, '
        'good baseline for mild noise.',
    ),
    (
        'Spectral',
        'Classic-spectral',
        'Spectral denoising using frequency-domain analysis. '
        'Effective for structured or periodic noise.',
    ),
    (
        'GM (Gaussian Mixture)',
        'Classic-gm',
        'Gaussian Mixture model-based denoising. Good for images '
        'with multimodal intensity distributions.',
    ),
    (
        'Auto (Classic)',
        None,
        'Automatically selects the best classical denoising method '
        'for the image. Fast but less powerful than ML methods.',
    ),
]


class AydinDenoiseWidget(QWidget):
    """Napari dock widget for quick image denoising with Aydin.

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        The napari viewer instance (injected by napari).
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._worker = None
        self._current_metadata = None

        self._init_ui()
        self._connect_events()
        self._refresh_layer_combo()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)

        # --- Image layer ---
        layer_group = QGroupBox('Input')
        layer_layout = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel('Image Layer:'))
        self._layer_combo = QComboBox()
        row.addWidget(self._layer_combo, 1)
        layer_layout.addLayout(row)
        layer_group.setLayout(layer_layout)
        layout.addWidget(layer_group)

        # --- Method ---
        method_group = QGroupBox('Method')
        method_layout = QVBoxLayout()
        self._method_combo = QComboBox()
        for i, (display_name, _, tooltip) in enumerate(_METHODS):
            self._method_combo.addItem(display_name)
            self._method_combo.setItemData(i, tooltip, role=Qt.ToolTipRole)
        method_layout.addWidget(self._method_combo)
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # --- Dimensions ---
        dim_group = QGroupBox('Dimensions')
        dim_layout = QVBoxLayout()

        self._axes_label = QLabel('Axes: (select a layer)')
        dim_layout.addWidget(self._axes_label)

        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel('Batch axes:'))
        self._batch_combo = QComboBox()
        self._batch_combo.addItem('None')
        batch_row.addWidget(self._batch_combo, 1)
        dim_layout.addLayout(batch_row)

        chan_row = QHBoxLayout()
        chan_row.addWidget(QLabel('Channel axes:'))
        self._chan_combo = QComboBox()
        self._chan_combo.addItem('None')
        chan_row.addWidget(self._chan_combo, 1)
        dim_layout.addLayout(chan_row)

        dim_group.setLayout(dim_layout)
        layout.addWidget(dim_group)

        # --- Action buttons ---
        btn_layout = QHBoxLayout()
        self._denoise_btn = QPushButton('Denoise')
        self._denoise_btn.clicked.connect(self._on_denoise_clicked)
        btn_layout.addWidget(self._denoise_btn)

        self._cancel_btn = QPushButton('Cancel')
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        btn_layout.addWidget(self._cancel_btn)
        layout.addLayout(btn_layout)

        # --- Studio launch ---
        self._studio_btn = QPushButton('Open Aydin Studio')
        self._studio_btn.clicked.connect(self._on_studio_clicked)
        layout.addWidget(self._studio_btn)

        layout.addStretch()
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Event wiring
    # ------------------------------------------------------------------

    def _connect_events(self):
        self._layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        self._viewer.layers.events.inserted.connect(self._refresh_layer_combo)
        self._viewer.layers.events.removed.connect(self._refresh_layer_combo)

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def _refresh_layer_combo(self, event=None):
        """Repopulate the layer combo with current Image layers."""
        import napari

        current_text = self._layer_combo.currentText()
        self._layer_combo.blockSignals(True)
        self._layer_combo.clear()
        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self._layer_combo.addItem(layer.name, layer)
        # Restore selection if possible
        idx = self._layer_combo.findText(current_text)
        if idx >= 0:
            self._layer_combo.setCurrentIndex(idx)
        self._layer_combo.blockSignals(False)
        self._on_layer_changed()

    def _get_selected_layer(self):
        """Return the currently selected napari Image layer or None."""
        idx = self._layer_combo.currentIndex()
        if idx < 0:
            return None
        return self._layer_combo.itemData(idx)

    # ------------------------------------------------------------------
    # Dimension detection
    # ------------------------------------------------------------------

    def _on_layer_changed(self, index=None):
        """Update dimension display when the selected layer changes."""
        layer = self._get_selected_layer()
        if layer is None:
            self._axes_label.setText('Axes: (select a layer)')
            self._current_metadata = None
            self._update_axis_combos(None)
            return

        metadata = detect_axes_from_napari_layer(layer, self._viewer)
        self._current_metadata = metadata
        self._axes_label.setText(f'Axes: {metadata.axes}  Shape: {metadata.shape}')
        self._update_axis_combos(metadata)

    def _update_axis_combos(self, metadata):
        """Populate batch/channel axis override combos from metadata."""
        for combo in (self._batch_combo, self._chan_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem('None')

        if metadata is not None:
            for i, code in enumerate(metadata.axes):
                label = f'{code} (dim {i}, size {metadata.shape[i]})'
                self._batch_combo.addItem(label, i)
                self._chan_combo.addItem(label, i)

            # Pre-select detected batch/channel axes
            if metadata.batch_axes:
                for i, is_b in enumerate(metadata.batch_axes):
                    if is_b:
                        idx = self._batch_combo.findData(i)
                        if idx >= 0:
                            self._batch_combo.setCurrentIndex(idx)
                        break

            if metadata.channel_axes:
                for i, is_c in enumerate(metadata.channel_axes):
                    if is_c:
                        idx = self._chan_combo.findData(i)
                        if idx >= 0:
                            self._chan_combo.setCurrentIndex(idx)
                        break

        for combo in (self._batch_combo, self._chan_combo):
            combo.blockSignals(False)

    def _get_batch_channel_axes(self):
        """Return (batch_axes, channel_axes) boolean tuples from current UI state."""
        metadata = self._current_metadata
        if metadata is None:
            return None, None

        ndim = len(metadata.shape)

        # Start from detected values
        batch_axes = (
            list(metadata.batch_axes) if metadata.batch_axes else [False] * ndim
        )
        channel_axes = (
            list(metadata.channel_axes) if metadata.channel_axes else [False] * ndim
        )

        # Override from combos if user selected specific axes
        batch_idx = self._batch_combo.currentData()
        if batch_idx is not None:
            batch_axes = [False] * ndim
            batch_axes[batch_idx] = True

        chan_idx = self._chan_combo.currentData()
        if chan_idx is not None:
            channel_axes = [False] * ndim
            channel_axes[chan_idx] = True

        return tuple(batch_axes), tuple(channel_axes)

    # ------------------------------------------------------------------
    # Denoising
    # ------------------------------------------------------------------

    def _on_denoise_clicked(self):
        """Launch denoising in a worker thread."""
        layer = self._get_selected_layer()
        if layer is None:
            try:
                from napari.utils.notifications import show_warning

                show_warning('No image layer selected.')
            except ImportError:
                pass
            return

        method_idx = self._method_combo.currentIndex()
        _, variant, _ = _METHODS[method_idx]
        batch_axes, channel_axes = self._get_batch_channel_axes()
        image = layer.data.copy()
        layer_name = layer.name
        layer_rgb = getattr(layer, 'rgb', False)

        self._denoise_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)

        method_name = self._method_combo.currentText()
        try:
            from napari.utils.notifications import show_info

            show_info(f'Aydin: denoising "{layer_name}" with {method_name}...')
        except ImportError:
            pass

        from napari.qt.threading import create_worker

        self._worker = create_worker(
            self._denoise_in_thread,
            image=image,
            variant=variant,
            batch_axes=batch_axes,
            channel_axes=channel_axes,
        )
        self._worker.returned.connect(
            lambda result: self._on_denoise_done(result, layer_name, layer_rgb)
        )
        self._worker.errored.connect(self._on_denoise_error)
        self._worker.finished.connect(self._on_denoise_finished)
        self._worker.start()

    @staticmethod
    def _denoise_in_thread(image, variant, batch_axes, channel_axes):
        """Run denoising (called in worker thread).

        Parameters
        ----------
        image : numpy.ndarray
            Image data to denoise.
        variant : str or None
            Denoiser variant string (e.g. 'Classic-butterworth').
            None means Classic with auto-selection.
        batch_axes : tuple of bool
            Boolean tuple indicating batch dimensions.
        channel_axes : tuple of bool
            Boolean tuple indicating channel dimensions.

        Returns
        -------
        numpy.ndarray
            Denoised image.
        """
        from napari.utils import progress

        from aydin.util.log.log import Log

        # Enable console output so Aydin logs appear in napari's console
        old_enable = Log.enable_output
        Log.enable_output = True

        pbr = progress(total=2, desc="Aydin: initialising...")
        try:
            if variant is None:
                from aydin.restoration.denoise.classic import Classic

                denoiser = Classic()
            else:
                from aydin.restoration.denoise.util.denoise_utils import (
                    get_denoiser_class_instance,
                )

                denoiser = get_denoiser_class_instance(variant=variant)

            pbr.set_description("Aydin: training...")
            denoiser.train(image, batch_axes=batch_axes, channel_axes=channel_axes)
            pbr.update(1)
            pbr.set_description("Aydin: denoising...")
            result = denoiser.denoise(
                image, batch_axes=batch_axes, channel_axes=channel_axes
            )
            pbr.update(1)
            pbr.set_description("Aydin: done")
            return result
        finally:
            pbr.close()
            Log.enable_output = old_enable

    def _on_denoise_done(self, result, layer_name, rgb=False):
        """Add denoised result as a new layer (runs on main thread)."""
        kwargs = {'name': f'{layer_name}_denoised'}
        if rgb:
            kwargs['rgb'] = True
        self._viewer.add_image(result, **kwargs)
        try:
            from napari.utils.notifications import show_info

            show_info(f'Aydin: done denoising "{layer_name}"')
        except ImportError:
            pass

    def _on_denoise_error(self, exc):
        """Show error notification."""
        try:
            from napari.utils.notifications import show_error

            show_error(f'Denoising failed: {exc}')
        except ImportError:
            pass

    def _on_denoise_finished(self):
        """Re-enable buttons after denoising completes or fails."""
        self._denoise_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._worker = None

    def _on_cancel_clicked(self):
        """Cancel a running denoise worker."""
        if self._worker is not None:
            self._worker.quit()
        self._cancel_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Studio launch
    # ------------------------------------------------------------------

    def _on_studio_clicked(self):
        """Open Aydin Studio or bring an existing window to front."""
        try:
            import aydin.napari_plugin._studio_bridge as bridge

            if bridge._studio_window is not None and bridge._studio_window.isVisible():
                bridge._studio_window.raise_()
                bridge._studio_window.activateWindow()
            else:
                bridge.launch_studio_from_napari(self._viewer)
        except Exception as exc:
            try:
                from napari.utils.notifications import show_error

                show_error(f'Failed to open Aydin Studio: {exc}')
            except Exception:
                pass
