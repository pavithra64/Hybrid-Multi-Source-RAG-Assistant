from pypdf import PdfReader


def load_pdf(path):

    text = ""

    reader = PdfReader(path)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def load_txt(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()