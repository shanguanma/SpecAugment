#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.python.framework import constant_op
from tensorflow.contrib.image import sparse_image_warp
import tensorflow as tf
time_warping_para=40
time_masking_para=50
frequency_masking_para=17
audio, sampling_rate = librosa.load("../data/61-70968-0002.wav") 
mel_spectrogram= librosa.feature.melspectrogram(y=audio,sr=sampling_rate,n_mels=40, hop_length=128, fmax=8000)
v = mel_spectrogram.shape[0]
tau = mel_spectrogram.shape[1]

# Step 1 : Time warping
# Image warping control point setting.
center_position = v/2 # center_position is int scale
random_point = np.random.randint(low=time_warping_para, high=tau - time_warping_para) # random_point is int scale
# warping distance chose.
w = np.random.uniform(low=0, high=time_warping_para) #w is int scale
# control_point_locations is list 
control_point_locations = [[center_position, random_point]]
# control_point_locations :<tf.Tensor  shape=(1, 1, 2) dtype=float32>
control_point_locations = constant_op.constant(
    np.float32(np.expand_dims(control_point_locations, 0)))


control_point_destination = [[center_position, random_point + w]]
# control_point_destination: <tf.Tensor  shape=(1, 1, 2) dtype=float32>
control_point_destination = constant_op.constant(
    np.float32(np.expand_dims(control_point_destination, 0)))

# mel spectrogram data type convert to tensor constant for sparse_image_warp.
mel_spectrogram = mel_spectrogram.reshape([1, mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1])
mel_spectrogram_op = constant_op.constant(np.float32(mel_spectrogram))
# mel_spectrogram_op :[batch, height, width, channels] float Tensor
# source_control_point_locations: [batch, num_control_points, 2] float Tensor
# dest_control_point_locations: [batch, num_control_points, 2] float Tensor
# interpolation_order: polynomial order used by the spline interpolation
# regularization_weight: weight on smoothness regularizer in interpolation
# num_boundary_points: How many zero-flow boundary points to include at each image edge.Usage: num_boundary_points=0: don't     # add zero-flow points num_boundary_points=1: 4 corners of the image num_boundary_points=2: 4 corners and one in the middle     # of each edge (8 points total) num_boundary_points=n: 4 corners and n-1 along each edge

# Note that image and offsets can be of type tf.half, tf.float32, or tf.float64, and do not necessarily have to be the same      type.
# Returns:
# warped_image: [batch, height, width, channels] float Tensor with same type as input image.
# flow_field: [batch, height, width, 2] float Tensor containing the dense flow field produced by the interpolation.
warped_mel_spectrogram_op, _ = sparse_image_warp(mel_spectrogram_op,
                                                 source_control_point_locations=control_point_locations,
                                                 dest_control_point_locations=control_point_destination,
                                                 interpolation_order=2,
                                                 regularization_weight=0,
                                                 num_boundary_points=1
                                                 )
print("warped_mel_spectrogram_op.shape before sess: ",warped_mel_spectrogram_op.shape)
print("warped_mel_spectrogram_op before sess: ",warped_mel_spectrogram_op)

# Change warp result's data type to numpy array for masking step.
with tf.Session() as sess:
    warped_mel_spectrogram = sess.run(warped_mel_spectrogram_op)

print("warped_mel_spectrogram.shape after sess: ",warped_mel_spectrogram.shape)
print("warped_mel_spectrogram after sess: ",warped_mel_spectrogram)

warped_mel_spectrogram = warped_mel_spectrogram.reshape([warped_mel_spectrogram.shape[1],
                                                         warped_mel_spectrogram.shape[2]])
print("warped_mel_spectrogram.shape after reshape: ",warped_mel_spectrogram.shape )
print("warped_mel_spectrogram after reshape: ",warped_mel_spectrogram)

