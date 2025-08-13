from PIL import Image, ImageOps
import numpy as np


def convert_to_mnist_format_with_centering(img):

    # 2. Invert colors (MNIST: black background, white digits)
    img = ImageOps.invert(img)

    # 3. Resize image so the longest side is 20 pixels (keep aspect ratio)
    img.thumbnail((20, 20), Image.LANCZOS)

    # 4. Create a 28x28 black canvas
    canvas = Image.new('L', (28, 28), color=0)
    canvas.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2))

    # 5. Convert to NumPy array for centroid calculation
    arr = np.array(canvas)

    # Sharpen image
    # threshold = 128
    # arr = (arr > threshold) * 255

    # 6. Calculate intensity-weighted centroid
    total = arr.sum()
    if total == 0:  # Avoid empty image
        center_y, center_x = 14, 14
    else:
        center_y, center_x = np.array(
            np.unravel_index(np.arange(arr.size), arr.shape)
        ).reshape(2, 28, 28)
        center_x = (arr * center_x).sum() / total
        center_y = (arr * center_y).sum() / total

    # 7. Calculate required shift
    shift_x = int(round(14 - center_x))
    shift_y = int(round(14 - center_y))

    # 8. Shift image
    shifted_img = Image.fromarray(np.roll(np.roll(arr, shift_y, axis=0), shift_x, axis=1))

    # 10. Return result
    return shifted_img

# Example:
# convert_to_mnist_format_with_centering("raw.png", "mnist_centered.png")
