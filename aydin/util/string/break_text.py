import re


def break_text(
    text, max_width: int = 80, clear_existing_breaks: bool = True, line_break='\n'
):

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


_html_clean_re = re.compile('<.*?>')


def _clean_html(text):
    if len(text.strip()) == 0:
        return text
    else:
        clean_text = text.replace('<a', '')
        clean_text = clean_text.replace('href=', '')
        clean_text = re.sub(r'https?:\/\/.*[\r\n]*', '', clean_text, flags=re.MULTILINE)
        clean_text = re.sub(_html_clean_re, '', clean_text)
        return clean_text
