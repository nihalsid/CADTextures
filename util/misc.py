from pathlib import Path


def read_list(path):
    path = Path(path)
    if path.exists():
        return [x.strip() for x in path.read_text().splitlines()]
    print(f'{path} does not exist, returning empty list.')
    return []


def write_list(path, listl):
    Path(path).write_text("\n".join(listl))


def print_model_parameter_count(model):
    from ballpark import business
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in {type(model).__name__}: {business(count, precision=3, prefix=True)}")
