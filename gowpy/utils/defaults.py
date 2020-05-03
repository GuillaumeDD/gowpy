from gowpy.gow.typing import Tokenized_document


def default_tokenizer(document: str) -> Tokenized_document:
    return document.split()
