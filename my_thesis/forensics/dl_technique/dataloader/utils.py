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