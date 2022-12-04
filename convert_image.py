# from PIL import Image

IMG_SIZE = 100


def prepare_image( image):

    ready_image = image.convert("L")
    ready_image = ready_image.resize((IMG_SIZE, IMG_SIZE))

    ready_image.thumbnail((IMG_SIZE, IMG_SIZE))
    return ready_image
