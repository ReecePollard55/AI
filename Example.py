import numpy as np
import skimage


def get_xy(tiff_file, type_):
    """
    Read, trim, and crop or down-sample TIFF file and reshape into two-dimensional
    data matrix, X, and target vector, y.

    :param tiff_file: Image file name (string)
    :param type_: 'crop' to crop the northwest corner of the image,
                  'ds' to down-sample by skipping every 10 pixels
    :return: data matrix 'X' (129600-by-2), target vector 'y' (129600-by-1)
    """
    # Read the image
    image = skimage.io.imread(tiff_file)

    # Trim the first and last 6 rows and columns
    image = image[6:-6, 6:-6]

    if type_ == 'crop':
        # crop to the first 360 rows and columns
        image = image[:360, :360]
    elif type_ == 'ds':
        # down-sample by keeping every 10th row and column
        image = image[::10, ::10]
    else:
        # if a bad type is used
        raise ValueError(f'Unknown test type: {type_}')

    # get the row index 'i' and column index 'j' for every pixel
    # (why start at -180 and not zero?)
    i, j = np.meshgrid(range(-180, 180), range(-180, 180), indexing='ij')

    # reshape row/column indexes into n-by-2 matrix
    x = np.concatenate((i.reshape((-1, 1)), j.reshape((-1, 1))), axis=1)

    # reshape elevations into n-by-1 vector
    y = image.reshape((-1, 1))

    return x, y



def error_bits(error):
    """
    Return a lower bound on the number of bits to encode the errors based on Shannon's source coding theorem.

    :param error: Vector or list of errors (error = estimate - actual)
    :return: The lower bound number of bits to encode the errors
    """
    # round and cast to an integer, reshape as a vector
    error = np.round(error).astype(int).reshape((-1,))

    # shift so that the minimum value is zero
    error = error - error.min()

    # count how many occurrences of each discrete value
    p = np.binount(error)

    # ignore zero counts
    p = p[p > 0]

    # convert counts into discrete probability distribution
    p = p / p.sum()

    # compute entropy (bits per codeword)
    entropy = -(p * np.log2(p)).sum()

    # minimum bits to encode all errors
    bits = int(np.ceil(entropy * len(error)))

    return bits