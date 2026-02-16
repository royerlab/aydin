"""Text wrapping and line-breaking utilities with HTML awareness."""

import re


def break_text(
    text, max_width: int = 80, clear_existing_breaks: bool = True, line_break='\n'
):
    """Wrap text to a maximum line width, with HTML tag awareness.

    Breaks text at word boundaries to fit within the specified width.
    Preserves double line breaks (paragraph separators). HTML tags
    are excluded from width calculations.

    Parameters
    ----------
    text : str
        Input text to wrap.
    max_width : int
        Maximum line width in characters.
    clear_existing_breaks : bool
        If True, removes existing single line breaks before re-wrapping,
        while preserving double line breaks.
    line_break : str
        Character(s) to use for line breaks.

    Returns
    -------
    str
        Wrapped text.
    """

    # We remove existing line breaks except double line breaks if requested:
    if clear_existing_breaks:
        text = text.replace('\n\n', '<!LB!>')
        text = text.replace('\n', ' ')
        text = text.replace('<!LB!>', '\n\n')

    # remove repeated spaces:
    text = re.sub(' +', ' ', text)

    # string_no_punctuation = re.sub("[^\w\s]", "", text)
    words = re.split(r'(\s+)', text)

    # words = re.findall(rf'[\w\W\-.",]+', text)

    output_text = ''
    counter = 0
    for word in words:
        if counter >= max_width or '\n' in word:
            output_text += line_break
            counter = 0

        if not (word == ' ' and counter == 0):
            if '\n' in word:
                word = word.strip() + '\n'
            output_text += word
            counter += len(_clean_html(word))

    return output_text


def strip_notgui(text: str) -> str:
    """Remove content after ``<notgui>`` for GUI display.

    Docstrings may contain a ``<notgui>`` marker that separates the
    GUI-visible portion (before the tag) from API-only documentation
    (after the tag). This function returns only the GUI-visible part.

    Parameters
    ----------
    text : str
        Docstring text, possibly containing a ``<notgui>`` tag.

    Returns
    -------
    str
        Text before the ``<notgui>`` tag (trailing whitespace stripped),
        or the full text if the tag is absent. Returns ``''`` for
        ``None`` or empty input.
    """
    if not text:
        return ''
    idx = text.find('<notgui>')
    if idx == -1:
        return text
    return text[:idx].rstrip()


_html_clean_re = re.compile('<.*?>')


def _clean_html(text):
    """Strip HTML tags and URLs from text for width calculation.

    Parameters
    ----------
    text : str
        Input text possibly containing HTML markup.

    Returns
    -------
    str
        Text with HTML tags and URLs removed.
    """
    if len(text.strip()) == 0:
        return text
    else:
        clean_text = text.replace('<a', '')
        clean_text = clean_text.replace('href=', '')
        clean_text = re.sub(r'https?:\/\/.*[\r\n]*', '', clean_text, flags=re.MULTILINE)
        clean_text = re.sub(_html_clean_re, '', clean_text)
        return clean_text
