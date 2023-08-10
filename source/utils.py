def get_args(d: dict, name: str) -> dict:
    return {k.split('_', 1)[-1]: v for k, v in d.items() if k.startswith('{}_'.format(name))}