import numpy as np
import torch


def create_circular_mask(h, w, center, radius):
    """
    Create a circular mask tensor.

    Args:
        h (int): The height of the mask tensor.
        w (int): The width of the mask tensor.
        center (Optional[Tuple[int, int]]): The center of the circle as a tuple (y, x). If None, the middle of the image is used.
        radius (Optional[int]): The radius of the circle. If None, the smallest distance between the center and image walls is used.

    Returns:
        A boolean tensor of shape [h, w] representing the circular mask.
    """
    if center is None:  # use the middle of the image
        center = (int(h / 2), int(w / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], h - center[0], w - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((Y - center[0]) ** 2 + (X - center[1]) ** 2)

    mask = dist_from_center <= radius
    mask = torch.from_numpy(mask).bool()
    return mask


def create_square_mask(
    height: int, width: int, center: list, radius: int
) -> torch.Tensor:
    """Create a square mask tensor.

    Args:
        height (int): The height of the mask.
        width (int): The width of the mask.
        center (list): The center of the square mask as a list of two integers. Order [y,x]
        radius (int): The radius of the square mask.

    Returns:
        torch.Tensor: The square mask tensor of shape (1, 1, height, width).

    Raises:
        ValueError: If the center or radius is invalid.
    """
    if not isinstance(center, list) or len(center) != 2:
        raise ValueError("center must be a list of two integers")
    if not isinstance(radius, int) or radius <= 0:
        raise ValueError("radius must be a positive integer")
    if (
        center[0] < radius
        or center[0] >= height - radius
        or center[1] < radius
        or center[1] >= width - radius
    ):
        raise ValueError("center and radius must be within the bounds of the mask")

    mask = torch.zeros((height, width), dtype=torch.float32)
    x1 = int(center[1]) - radius
    x2 = int(center[1]) + radius
    y1 = int(center[0]) - radius
    y2 = int(center[0]) + radius
    mask[y1 : y2 + 1, x1 : x2 + 1] = 1.0
    return mask.bool()