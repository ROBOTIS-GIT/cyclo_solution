def display_to_source(x, y, display_size, source_size):
    """Return the source pixel (u, v) under a click on the scaled display; integer = center."""
    display_width, display_height = display_size
    source_width, source_height = source_size
    if display_width <= 0 or display_height <= 0:
        raise ValueError(f'invalid display size: {display_size}')
    if not (0 <= x < display_width and 0 <= y < display_height):
        raise ValueError(f'click ({x}, {y}) is outside the displayed image {display_size}')
    u = min(int(x * source_width / display_width), source_width - 1)
    v = min(int(y * source_height / display_height), source_height - 1)
    return float(u), float(v)
