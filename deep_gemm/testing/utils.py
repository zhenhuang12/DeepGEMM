import functools
import os
import re
import torch
from typing import Callable


def get_device_arch() -> str:
    if not torch.version.hip:
        major, minor = torch.cuda.get_device_capability()
        return f"sm{major}{minor}"

    env_arch = os.getenv('TORCH_CUDA_ARCH_LIST', '').strip().lower()
    if env_arch:
        return env_arch.split(',')[0].strip()

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    for attr in ('gcnArchName', 'gcn_arch_name', 'arch'):
        value = getattr(props, attr, None)
        if isinstance(value, str):
            match = re.search(r'(gfx\d+)', value.lower())
            if match:
                return match.group(1)

    device_name = torch.cuda.get_device_name(torch.cuda.current_device()).lower()
    if 'mi350' in device_name or 'mi355' in device_name:
        return 'gfx950'
    if 'mi300' in device_name or 'mi325' in device_name:
        return 'gfx942'
    return 'gfx942'


def get_arch_major() -> int:
    if not torch.version.hip:
        major, minor = torch.cuda.get_device_capability()
        return major

    arch = get_device_arch()
    if arch.startswith('gfx95'):
        return 10
    if arch.startswith('gfx94'):
        return 9
    match = re.match(r'gfx(\d+)', arch)
    return int(match.group(1)[0]) if match else 9


def test_filter(condition: Callable):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if condition():
                func(*args, **kwargs)
            else:
                print(f'{func.__name__}:')
                print(f' > Filtered by {condition}')
                print()
        return wrapper
    return decorator


def ignore_env(name: str, condition: Callable):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if condition():
                saved = os.environ.pop(name, None)
                func(*args, **kwargs)
                if saved is not None:
                    os.environ[name] = saved
            else:
                func(*args, **kwargs)
                
        return wrapper
    return decorator
