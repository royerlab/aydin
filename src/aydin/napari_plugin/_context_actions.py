"""Layer list context menu actions for one-click denoising."""

_registered = False
_dispose_callbacks = []


def register_context_actions(ctx=None):
    """Register 'Denoise' actions in napari's layer list right-click menu.

    Safe to call multiple times — actions are only registered once.
    Uses napari's app_model to add two items:

    - **Denoise (high quality)** — N2S-FGR CatBoost
    - **Denoise (fast)** — Classic Butterworth

    Parameters
    ----------
    ctx : npe2.PluginContext, optional
        Plugin context passed by npe2 when used as ``on_activate`` hook.
    """
    global _registered
    if _registered:
        return
    _registered = True

    try:
        from app_model.types import Action
        from napari._app_model import get_app_model
        from napari._app_model.constants import MenuGroup, MenuId
        from napari.components import LayerList
    except ImportError:
        return  # napari version too old or app_model not available

    app = get_app_model()

    def _denoise_selected_layer(variant, quality_label, ll):
        """Denoise the active layer in a background thread."""
        import napari.layers

        layer = ll.selection.active
        if layer is None or not isinstance(layer, napari.layers.Image):
            return

        image = layer.data.copy()
        layer_name = layer.name
        layer_rgb = getattr(layer, 'rgb', False)

        from napari import current_viewer

        viewer = current_viewer()
        if viewer is None:
            return

        # Detect axes on main thread (viewer properties are not thread-safe)
        from aydin.napari_plugin._axes_utils import detect_axes_from_napari_layer

        metadata = detect_axes_from_napari_layer(layer, viewer)
        batch_axes = metadata.batch_axes
        channel_axes = metadata.channel_axes

        try:
            from napari.utils.notifications import show_info

            show_info(f'Aydin: denoising "{layer_name}" ({quality_label})...')
        except ImportError:
            pass

        from napari.qt.threading import create_worker

        def _run():
            from napari.utils import progress

            from aydin.util.log.log import Log

            old_enable = Log.enable_output
            Log.enable_output = True
            pbr = progress(total=2, desc=f"Aydin: {quality_label}...")
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

        worker = create_worker(_run)

        def _on_done(result):
            kwargs = {'name': f'{layer_name}_{quality_label}'}
            if layer_rgb:
                kwargs['rgb'] = True
            viewer.add_image(result, **kwargs)
            try:
                from napari.utils.notifications import show_info

                show_info(f'Aydin: done denoising "{layer_name}"')
            except ImportError:
                pass

        worker.returned.connect(_on_done)
        worker.errored.connect(lambda exc: _show_error(f'Denoising failed: {exc}'))
        worker.start()

    def _show_error(msg):
        try:
            from napari.utils.notifications import show_error

            show_error(msg)
        except ImportError:
            pass

    # Properly annotated callbacks so in_n_out can inject LayerList.
    # Cannot use lambdas — in_n_out needs real type annotations to
    # resolve the dependency injection.

    def _denoise_hq(ll: LayerList):
        _denoise_selected_layer('Noise2SelfFGR-cb', 'denoised_hq', ll)

    def _denoise_fast(ll: LayerList):
        _denoise_selected_layer('Classic-butterworth', 'denoised_fast', ll)

    menu_kwargs = {
        'id': MenuId.LAYERLIST_CONTEXT,
        'group': MenuGroup.LAYERLIST_CONTEXT.SPLIT_MERGE,
    }

    dispose = app.register_action(
        Action(
            id='aydin.denoise_high_quality',
            title='Denoise (high quality)',
            callback=_denoise_hq,
            menus=[menu_kwargs],
        )
    )
    _dispose_callbacks.append(dispose)

    dispose = app.register_action(
        Action(
            id='aydin.denoise_fast',
            title='Denoise (fast)',
            callback=_denoise_fast,
            menus=[menu_kwargs],
        )
    )
    _dispose_callbacks.append(dispose)


def unregister_context_actions():
    """Remove the context menu actions (for cleanup)."""
    global _registered
    for dispose in _dispose_callbacks:
        dispose()
    _dispose_callbacks.clear()
    _registered = False
