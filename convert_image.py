from PIL import Image

IMG_SIZE = 100


def prepare_image(xmin, ymin, xmax, ymax, image):
    width = xmax - xmin
    height = ymax - ymin

    if width < height:
        croped_image = image.crop((xmin, ymin, xmax, ymin + width))
    else:
        croped_image = image.crop((xmin, ymin, xmin + height, ymax))

    ready_image = croped_image.convert("L")

    if ready_image.width < IMG_SIZE or ready_image.height < IMG_SIZE:
        ready_image = ready_image.resize((IMG_SIZE, IMG_SIZE))

    ready_image.thumbnail((IMG_SIZE, IMG_SIZE))
    return ready_image
