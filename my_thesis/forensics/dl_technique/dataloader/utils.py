import numpy as np

"""
    Make weight for each image.
    @param images: List(tuple(str, int)). Eg: [(img_path, class_label), ...]
    @param nclasses: int. Number of classes. Eg: 2
    @return weight: List(float). Weight for each image in <images>  (len(weight) == len(images))
    @info: calculation method:
        - Calculate <count> (number of images for each class).  eg: count[class_0] = N1, count[class_1] = N2
        - Calculate <weight_per_class>.                         eg: weight_class_0 = (N1+N2)/N1, weight_class_1 = (N1+N2)/N2 (the larger samples(count) is, the smaller count is
        - Assign weight for each image of each class.           eg: weight[<index_of_image_in_class0>] = weight_class_0

"""
def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    print(count)
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    print(weight_per_class)
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight, count

def make_weights_for_balanced_classes_2(image_paths, nclasses):
    def find_label(path: str):
        if '/0_real/' in path:
            return 0
        return 1

    count = [0] * nclasses
    for p in image_paths:
        count[find_label(p)] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    print(count)
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    print(weight_per_class)
    weight = [0] * len(image_paths)
    for idx, p in enumerate(image_paths):
        weight[idx] = weight_per_class[find_label(p)]
    return weight, count

def azimuthalAverage(magnitude_image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
            None, which then uses the center of the image (including
            fracitonal pixels).

    """
    # Calculate the indices from the image
    y, x = np.indices(magnitude_image.shape)

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = magnitude_image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr
    return radial_prof