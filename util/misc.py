from pathlib import Path


def read_list(path):
    return [x.strip() for x in Path(path).read_text().splitlines()]


def write_list(path, listl):
    Path(path).write_text("\n".join(listl))
