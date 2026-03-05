"""Time formatting helpers."""


def format_duration(seconds):
    """Execute `format_duration`.

    Args:
        seconds: Input parameter `seconds`.

    Returns:
        Any: Output produced by this function.
    """
    total = int(max(0.0, float(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
