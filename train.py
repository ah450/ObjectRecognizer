import descriptor
import bow
import os
from multiprocessing import Pool

def gen_dir(directory):
    """
    Generate descriptors for all files in a dirctory 
    (non recursive, doesn't even check for child directories)
    """
    for file in os.listdir(directory):
        if not '.sift' in file and not '.bow' in file: 
            descriptor.process(os.path.join(directory, file))


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
    # Find means
    files = descriptor.select_sample(cars_dir, cows_dir, bikes_dir)
    centers = bow.kmeans(files)
    # Calculate bow for every image
    results = p.map_async(bow_proxy, [(cars_dir, centers), (cows_dir, centers), (bikes_dir, centers)], 1)
    results.get()
    p.close()
    p.join()
    