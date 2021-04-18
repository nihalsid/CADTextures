from pathlib import Path


def read_list(path):
    return [x.strip() for x in Path(path).read_text().splitlines()]
