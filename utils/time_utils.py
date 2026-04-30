"""
Time formatting helpers — chuyển elapsed seconds thành human-readable string.

Dùng để hiển thị epoch time, ETA và tổng thời gian training trong train.py
và seed_sweep.py.
"""


def format_duration(seconds):
    """Chuyển đổi số giây thành chuỗi HH:MM:SS hoặc MM:SS.

    Args:
        seconds: Số giây (float hoặc int); giá trị âm được treat as 0.

    Returns:
        str: Dạng "HH:MM:SS" nếu >= 1 giờ, ngược lại "MM:SS".
             Ví dụ: 3661 → "01:01:01", 125 → "02:05".
    """
    total = int(max(0.0, float(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
