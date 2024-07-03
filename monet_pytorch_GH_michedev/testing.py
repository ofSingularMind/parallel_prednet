# from multi_object_datasets import multi_dsprites
# import tensorflow as tf
# import torch

# # Function to parse a single example from the TFRecords file
# def _parse_function(proto):
#     # Define your features here. Example:
#     feature_description = {
#         'color': tf.io.FixedLenFeature([], tf.float32),
#         'image': tf.io.FixedLenFeature([], tf.string),
#         'mask': tf.io.FixedLenFeature([], tf.string),
#         'orientation': tf.io.FixedLenFeature([], tf.float32),
#         'scale': tf.io.FixedLenFeature([], tf.float32),
#         'shape': tf.io.FixedLenFeature([], tf.float32),
#         'visibility': tf.io.FixedLenFeature([], tf.float32),
#         'x': tf.io.FixedLenFeature([], tf.float32),
#         'y': tf.io.FixedLenFeature([], tf.float32),
#     }
#     parsed_features = tf.io.parse_single_example(proto, feature_description)
#     return parsed_features['image']

# # Load TFRecords file
# def load_tfrecords(tfrecords_file):
#     raw_dataset = tf.data.TFRecordDataset(tfrecords_file)
#     parsed_dataset = raw_dataset.map(_parse_function)
#     return parsed_dataset

# # Example usage
# tfrecords_file = '/home/evalexii/Documents/Thesis/code/parallel_prednet/monet_pytorch_GH_michedev/data/multi_dsprites_colored_on_colored.tfrecords'
# parsed_dataset = load_tfrecords(tfrecords_file)

# from torch.utils.data import Dataset

# class TFRecordDataset(Dataset):
#     def __init__(self, tf_dataset):
#         self.tf_dataset = tf_dataset
#         self.data = list(tf_dataset.as_numpy_iterator())

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         image = self.data[idx]
#         image = torch.from_numpy(image).float()
#         label = torch.tensor(label, dtype=torch.long)
#         return image, label

# # Convert the parsed TFRecords dataset to a PyTorch dataset
# pytorch_dataset = TFRecordDataset(parsed_dataset)

# print("ho")

# tf_records_path = '/home/evalexii/Documents/Thesis/code/parallel_prednet/monet_pytorch_GH_michedev/data/multi_dsprites_colored_on_colored.tfrecords'
# batch_size = 32

# dataset = multi_dsprites.dataset(tf_records_path, 'colored_on_colored')
# batched_dataset = dataset.batch(batch_size)  # optional batching
# iterator = batched_dataset.make_one_shot_iterator()
# data = iterator.get_next()

# # with tf.train.SingularMonitoredSession() as sess:
#     # d = sess.run(data)

# from torchdata.datapipes.iter import FileLister, FileOpener
# datapipe1 = FileLister("/home/evalexii/Documents/Thesis/code/parallel_prednet/monet_pytorch_GH_michedev/data", "*.tfrecords", recursive=True)
# print(datapipe1.length)
# datapipe2 = FileOpener(datapipe1, mode="b")
# tfrecord_loader_dp = datapipe2.load_from_tfrecord()
# for example in tfrecord_loader_dp:
#     print(example)


import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.backends.cudnn.version())
