"""Device selection utilities."""

import torch


def resolve_device(preferred='auto'):
    """
    Resolve torch device from preference.

    Preference order:
    - 'auto': cuda -> mps -> cpu
    - 'cuda'/'mps': use if available, otherwise fallback to auto order
    - 'cpu': always cpu
    """
    preferred = (preferred or 'auto').lower()

    if preferred == 'cpu':
        return torch.device('cpu')

    if preferred == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')

    if preferred == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')

    if torch.cuda.is_available():
        return torch.device('cuda')

    if torch.backends.mps.is_available():
        return torch.device('mps')

    return torch.device('cpu')
