"""CLI styling utilities for Aydin.

Provides colored output helpers, a custom Click HelpFormatter, and the
ASCII art banner used by the Aydin command-line interface.  All styling
uses Click's built-in ``click.style()`` which degrades gracefully on
terminals that do not support ANSI colors.
"""

import click

from aydin import __version__

# ── Unicode symbols ─────────────────────────────────────────────────
SYM_CHECK = '\u2713'  # ✓
SYM_BULLET = '\u00bb'  # »
SYM_H = '\u2500'  # ─  horizontal box line
SYM_TL = '\u250c'  # ┌  top-left corner
SYM_TR = '\u2510'  # ┐  top-right corner
SYM_BL = '\u2514'  # └  bottom-left corner
SYM_BR = '\u2518'  # ┘  bottom-right corner
SYM_V = '\u2502'  # │  vertical line

# ── ASCII banner ────────────────────────────────────────────────────
_BANNER = r"""
    _             _ _
   / \  _   _  __| (_)_ __
  / _ \| | | |/ _` | | '_ \
 / ___ \ |_| | (_| | | | | |
/_/   \_\__, |\__,_|_|_| |_|
        |___/"""


def styled_banner():
    """Return the Aydin banner with version, styled with colors."""
    lines = _BANNER.strip('\n').splitlines()
    styled = [click.style(line, fg='cyan', bold=True) for line in lines]
    version = click.style(f'  v{__version__}', fg='cyan')
    tagline = click.style('  Self-supervised image denoising', dim=True)
    styled.append(version + tagline)
    return '\n'.join(styled)


# ── Helper formatters ───────────────────────────────────────────────


def success_message(text):
    """Format a success message with a green checkmark."""
    return click.style(f' {SYM_CHECK} ', fg='green', bold=True) + click.style(
        text, fg='green'
    )


def styled_metric(name, value):
    """Format a metric name-value pair with colors."""
    return (
        click.style(name, fg='cyan', bold=True)
        + click.style(': ', dim=True)
        + click.style(f'{value}', fg='white', bold=True)
    )


# ── Custom HelpFormatter ───────────────────────────────────────────


class AydinHelpFormatter(click.HelpFormatter):
    """Click HelpFormatter subclass with colored headings and styled entries."""

    def write_usage(self, prog, args='', prefix=None):
        if prefix is None:
            prefix = click.style('Usage: ', fg='yellow', bold=True)
        super().write_usage(prog, args, prefix=prefix)

    def write_heading(self, heading):
        styled = click.style(f'{heading}:', fg='yellow', bold=True)
        self.write(f"{'':>{self.current_indent}}{styled}\n")

    def write_dl(self, rows, col_max=30, col_spacing=2):
        styled_rows = []
        for first, second in rows:
            if first.lstrip().startswith('-'):
                styled_first = click.style(first, fg='cyan')
            else:
                styled_first = click.style(first, fg='green')
            styled_rows.append((styled_first, second))
        super().write_dl(styled_rows, col_max=col_max, col_spacing=col_spacing)


class AydinContext(click.Context):
    """Custom Click Context that uses :class:`AydinHelpFormatter`."""

    def make_formatter(self):
        return AydinHelpFormatter(
            width=self.terminal_width, max_width=self.max_content_width
        )


# ── Denoiser listing ───────────────────────────────────────────────

DEFAULT_DENOISER = 'Noise2SelfFGR-cb'

_CATEGORY_INFO = {
    'Classic': 'Classical signal processing methods',
    'Noise2SelfCNN': 'Self-supervised CNN architectures',
    'Noise2SelfFGR': 'Feature generation & regression',
}

# Short, unique descriptions per variant (the combined docstrings from
# the restoration modules are too long/generic for a CLI listing).
_VARIANT_DESCRIPTIONS = {
    # Classic denoisers
    'Classic-butterworth': 'Low-pass frequency filter (fast, good baseline)',
    'Classic-dictionary_fixed': 'Sparse coding with a fixed dictionary',
    'Classic-dictionary_learned': 'Sparse coding with a learned dictionary',
    'Classic-gaussian': 'Gaussian blur smoothing filter',
    'Classic-gm': 'Gaussian mixture model denoiser',
    'Classic-harmonic': 'Harmonic/spectral frequency filter',
    'Classic-lipschitz': 'Lipschitz continuity regularization',
    'Classic-nlm': 'Non-local means patch-based averaging',
    'Classic-pca': 'PCA-based patch denoising',
    'Classic-spectral': 'Spectral filtering in frequency domain',
    'Classic-tv': 'Total variation regularization',
    'Classic-wavelet': 'Wavelet shrinkage / thresholding',
    # CNN models
    'Noise2SelfCNN-dncnn': 'DnCNN feed-forward network (Zhang et al.)',
    'Noise2SelfCNN-jinet': 'J-invariant dilated convolution network',
    'Noise2SelfCNN-linear_scaling_unet': 'UNet with linear filter scaling',
    'Noise2SelfCNN-res_unet': 'UNet with additive skip connections',
    'Noise2SelfCNN-ronneberger_unet': 'Original Ronneberger UNet architecture',
    'Noise2SelfCNN-unet': 'Standard UNet encoder-decoder',
    # FGR regressors
    'Noise2SelfFGR-cb': 'CatBoost gradient boosting (fast, GPU)',
    'Noise2SelfFGR-lgbm': 'LightGBM gradient boosting',
    'Noise2SelfFGR-linear': 'Linear regression',
    'Noise2SelfFGR-perceptron': 'Multi-layer perceptron',
    'Noise2SelfFGR-random_forest': 'Random forest ensemble',
    'Noise2SelfFGR-support_vector': 'Support vector regression',
}


def format_denoiser_listing(names, descriptions):
    """Format ``--list-denoisers`` output with colors and category grouping.

    Parameters
    ----------
    names : list of str
        Denoiser variant names (e.g. ``'Classic-butterworth'``).
    descriptions : list of str
        Raw HTML descriptions (unused; variant descriptions come from
        the static ``_VARIANT_DESCRIPTIONS`` map for conciseness).

    Returns
    -------
    str
        Formatted, colored listing.
    """
    groups = {}
    for name in names:
        category = name.split('-', 1)[0]
        if category not in groups:
            groups[category] = []
        groups[category].append(name)

    lines = [
        click.style('Available denoiser variants', fg='cyan', bold=True),
        '',
    ]

    for category, variant_names in groups.items():
        cat_desc = _CATEGORY_INFO.get(category, '')
        lines.append(
            '  '
            + click.style(category, fg='yellow', bold=True)
            + click.style(f'  {cat_desc}', dim=True)
        )
        for full_name in variant_names:
            desc = _VARIANT_DESCRIPTIONS.get(full_name, '')
            is_default = full_name == DEFAULT_DENOISER
            name_styled = click.style(full_name, fg='green', bold=is_default)
            parts = '    ' + click.style(SYM_BULLET, fg='cyan') + ' ' + name_styled
            if desc:
                parts += click.style(f'  {desc}', dim=True)
            if is_default:
                parts += click.style('  (default)', fg='yellow')
            lines.append(parts)
        lines.append('')

    return '\n'.join(lines)


# ── Citation box ────────────────────────────────────────────────────


def format_cite_box():
    """Return a styled citation box for the ``cite`` command."""
    doi = 'https://doi.org/10.5281/zenodo.5654826'
    width = 56

    c = 'cyan'
    top = click.style(SYM_TL + SYM_H * (width - 2) + SYM_TR, fg=c)
    bot = click.style(SYM_BL + SYM_H * (width - 2) + SYM_BR, fg=c)
    s = click.style(SYM_V, fg=c)

    def row(text='', raw_len=None):
        if raw_len is None:
            raw_len = len(text)
        pad = width - 2 - raw_len
        lp = pad // 2
        rp = pad - lp
        return s + ' ' * lp + text + ' ' * rp + s

    title = click.style('Citing Aydin', fg='cyan', bold=True)
    msg = 'If you find Aydin useful, please cite:'
    link = click.style(doi, fg='blue', underline=True)

    return '\n'.join(
        [
            '',
            top,
            row(),
            row(title, raw_len=12),
            row(),
            row(msg, raw_len=len(msg)),
            row(),
            row(link, raw_len=len(doi)),
            row(),
            bot,
            '',
        ]
    )


# ── Command sections (themed grouping for --help) ─────────────────

_COMMAND_SECTIONS = [
    ('Denoising', ['denoise', 'benchmark-algos']),
    ('Metrics', ['ssim', 'psnr', 'mse', 'fsc']),
    ('Image Tools', ['info', 'view', 'split-channels', 'hyperstack']),
    ('About', ['cite']),
]


# ── Click Group subclass ───────────────────────────────────────────


class AydinCommand(click.Command):
    """Click Command subclass that uses :class:`AydinContext`."""

    context_class = AydinContext


class AydinGroup(click.Group):
    """Click Group that prepends the Aydin banner and uses styled formatting.

    Commands are displayed in themed sections rather than alphabetical
    order.  The grouping is defined by :data:`_COMMAND_SECTIONS`.
    """

    context_class = AydinContext

    def command(self, *args, **kwargs):
        """Override to inject :class:`AydinCommand` as the default cls."""
        kwargs.setdefault('cls', AydinCommand)
        return super().command(*args, **kwargs)

    def format_help(self, ctx, formatter):
        formatter.write(styled_banner())
        formatter.write('\n\n')
        super().format_help(ctx, formatter)

    def format_commands(self, ctx, formatter):
        """Write commands grouped by theme instead of alphabetically."""
        commands = {}
        for subcommand in self.list_commands(ctx):
            cmd = self.get_command(ctx, subcommand)
            if cmd is None or cmd.hidden:
                continue
            commands[subcommand] = cmd

        if not commands:
            return

        limit = formatter.width - 6 - max(len(n) for n in commands)

        # Render each themed section
        for section_name, section_cmds in _COMMAND_SECTIONS:
            rows = []
            for name in section_cmds:
                cmd = commands.pop(name, None)
                if cmd is None:
                    continue
                help_text = cmd.get_short_help_str(limit=limit)
                rows.append((name, help_text))

            if rows:
                with formatter.section(section_name):
                    formatter.write_dl(rows)

        # Any commands not listed in _COMMAND_SECTIONS go in "Other"
        if commands:
            rows = [
                (name, commands[name].get_short_help_str(limit=limit))
                for name in sorted(commands)
            ]
            with formatter.section('Other'):
                formatter.write_dl(rows)
