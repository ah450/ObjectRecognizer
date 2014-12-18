import descriptor
import bow
import os
from multiprocessing import Pool
import kdtree
from scipy import spatial

def gen_dir(directory):
    """
    Generate descriptors for all files in a dirctory 
    (non recursive, doesn't even check for child directories)
    """
    for file in os.listdir(directory):
        if not '.sift' in file and not '.bow' in file: 
            descriptor.process(os.path.join(directory, file))


class CentroidWrapper(list):
    def __init__(self, centroid, cluster):
        super(CentroidWrapper, self).__init__(centroid)
        self.cluster = cluster




def bow_proxy(tu):
    return bow.bow(*tu)

if __name__ == "__main__":
    cars_dir = "PNGImages/cars"
    cows_dir = "PNGImages/cows"
    bikes_dir = "PNGImages/bikes"
    # Generate Descriptors
    p = Pool(3)
    results = p.map_async(gen_dir, [cars_dir, cows_dir, bikes_dir], 1)
    results.get()
    # # Find means
    files = descriptor.select_sample(cars_dir, cows_dir, bikes_dir)
    centers = bow.kmeans(files)
    wrapped_centers = [CentroidWrapper(c.tolist(), i) for i, c in enumerate(centers)]
    # Construct kd-tree
    tree = kdtree.create(point_list=wrapped_centers, dimensions=128)
    if not tree.is_balanced:
        tree = tree.rebalance()
    p.close()
    p.join()
    # Calculate bow for every image
    bow.bow(cars_dir, tree)
    bow.bow(cows_dir, tree)
    bow.bow(bikes_dir, tree)
 
    