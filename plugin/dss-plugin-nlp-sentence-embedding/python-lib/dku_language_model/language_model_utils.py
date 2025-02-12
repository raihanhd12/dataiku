# -*- coding: utf-8 -*-
import logging
import re
import numpy as np
import sys
if sys.version_info[0] >= 3:
    unicode = str

try:
    import string
    # Python 2
    maketrans = string.maketrans
except AttributeError:
    # Python 3
    maketrans = str.maketrans

logger = logging.getLogger(__name__)

# Twitter related tokens
# Py2 <> Py3 compatibility via br''.decode('raw_unicode_escape') (https://stackoverflow.com/a/42924286)
RE_HASHTAG = br'#[a-zA-Z0-9_]+'.decode('raw_unicode_escape')
RE_MENTION = br'@[a-zA-Z0-9_]+'.decode('raw_unicode_escape')

RE_URL = br'(?:https?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'.decode('raw_unicode_escape')
RE_EMAIL = br'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b'.decode('raw_unicode_escape')

TOKENS_TO_IGNORE = [
    RE_HASHTAG,
    RE_MENTION,
    RE_URL,
    RE_EMAIL
]


def clean_text(text):
    """
    Applies some pre-processing to clean text data.

    In particular:
    - lowers the string
    - removes URLs, e-mail adresses
    - removes Twitter mentions and hastags
    - removes HTML tags
    - removes the character [']
    - replaces punctuation with spaces

    """
    if text is np.nan:
        return ''
    text = str(text).lower()  # lower text

    # ignore urls, mails, twitter mentions and hashtags
    for regex in TOKENS_TO_IGNORE:
        text = re.sub(regex, ' ', text)
    text = re.sub(r'<[^>]*>', ' ', text)  # remove HTML tags if any

    # remove the character [']
    text = re.sub(r"\'", "", text)

    # this is the default cleaning in Keras,
    # it consists in lowering the texts and removing the punctuation
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    split = " "  # character that will be used to split the texts later

    if isinstance(text, unicode):
        translate_map = dict((ord(c), unicode(split)) for c in filters)
        text = text.translate(translate_map)
    elif len(split) == 1:
        translate_map = maketrans(filters, split * len(filters))
        text = text.translate(translate_map)
    else:
        for c in filters:
            text = text.replace(c, split)
    return text
