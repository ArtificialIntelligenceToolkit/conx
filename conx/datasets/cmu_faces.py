import conx as cx
import numpy as np

def cmu_faces_full_size(*args, path="cmu_faces_full_size.npz", **kwargs):
    dataset = cx.Dataset()
    inputs, labels = load_dataset_npz(
        path,
        "https://raw.githubusercontent.com/Calysto/conx-data/master/cmu_faces/cmu_faces_full_size.npz")
    dataset.name = "CMU Faces, full-size"
    dataset.description = """
Original source: http://archive.ics.uci.edu/ml/datasets/cmu+face+images
"""
    return process_face_data(dataset, inputs, labels)

def cmu_faces_quarter_size(*args, path="cmu_faces_quarter_size.npz", **kwargs):
    dataset = cx.Dataset()
    inputs, labels = load_dataset_npz(
        path,
        "https://raw.githubusercontent.com/Calysto/conx-data/master/cmu_faces/cmu_faces_quarter_size.npz")
    dataset.name = "CMU Faces, quarter-size"
    dataset.description = """
Original source: http://archive.ics.uci.edu/ml/datasets/cmu+face+images
"""
    return process_face_data(dataset, inputs, labels)

def cmu_faces_half_size(*args, path="cmu_faces_half_size.npz", **kwargs):
    dataset = cx.Dataset()
    inputs, labels = load_dataset_npz(
        path,
        "https://raw.githubusercontent.com/Calysto/conx-data/master/cmu_faces/cmu_faces_half_size.npz")
    dataset.name = "CMU Faces, half-size"
    dataset.description = """
Original source: http://archive.ics.uci.edu/ml/datasets/cmu+face+images
"""
    return process_face_data(dataset, inputs, labels)

def load_dataset_npz(path, url):
    """loads an .npz file of saved image data, and returns the images and their
    associated labels as numpy arrays
    """
    from keras.utils import get_file
    path = get_file(path, origin=url)
    f = np.load(path, allow_pickle=True)
    images, labels = f['data'], f['labels']
    return images, labels

def create_pose_targets(labels):
    """converts a list of label strings to one-hot pose target vectors"""
    pose_names = ['left', 'forward', 'up', 'right']
    make_target_vector = lambda x: [int(x == name) for name in pose_names]
    poses = [s.split('_')[1] for s in labels]
    return np.array([make_target_vector(p) for p in poses]).astype('uint8')

def process_face_data(dataset, inputs, labels):
    targets = create_pose_targets(labels)
    dataset.load_direct([inputs], [targets], [labels])
    return dataset
