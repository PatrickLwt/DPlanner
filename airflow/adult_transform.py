
import tensorflow as tf 
import tensorflow_transform as tft
import numpy as np
import sys
import noise_utils
import tensorflow_probability as tfp

noise_type = 0
feature_num = 14

def sparse_to_dense(value):
    if isinstance(value, tf.sparse.SparseTensor):
        return tf.sparse.to_dense(value)
    return value

def transformed_name(key):
    return key + '_xf'

def process_block(inputs):
    feature, epsilon, block = inputs
    tf.print("Blockkkkk: ", block)
    tf.print("Epsilonnn: ", epsilon)
    tf.print("Feature: ", feature)
    feature = tf.reshape(feature, [-1, feature_num])
    
    # Calculate the sensitivity of this task
    sensitivity = 1.0 * feature_num
    
    # add noise to the training data blocks
    is_eval = tf.math.equal(block, tf.constant(0, dtype=tf.int64))   
    def not_process(x):
        return x
    
    feature_processed = tf.cond(is_eval, lambda: not_process(feature), lambda: noise_utils.laplace_noise_input(feature, sensitivity, epsilon))

    return feature_processed
    

def preprocessing_fn(inputs, custom_config):
    outputs = {}
    
    # Read the epsilon allocation result
    eps_array = np.loadtxt(custom_config)
    block_list = tf.cast(tf.convert_to_tensor(eps_array[0,:]), tf.int64)
    eps_list = tf.convert_to_tensor(eps_array[1,:])
    
    # Whether this block in the chosen block list
    blocks = sparse_to_dense(inputs['block'])
    equal = tf.math.equal(blocks, block_list)
    
    def get_epsilon(eps_idx):
        idx = tf.where(eps_idx)
        epsilon = eps_list[idx[0][0]]
        return tf.cast(epsilon, tf.float32)
    
    epsilon = tf.vectorized_map(get_epsilon, equal)
    tf.print("Epsilon: ", epsilon)
    dense = sparse_to_dense(inputs['feature'])
    print("Dense: ", dense)
    
    if noise_type == 0:
        noised = tf.map_fn(process_block, (dense, epsilon, blocks), fn_output_signature=tf.float32)
        outputs[transformed_name('feature')] = tf.reshape(noised, [-1, feature_num])
    else:
        reshape = tf.reshape(dense, [-1, feature_num])
        outputs[transformed_name('feature')] = reshape
    
    label_dense = sparse_to_dense(inputs['label'])
    outputs[transformed_name('label')] = tf.reshape(label_dense, [-1,])
    
    def assign_epsilon(inputs):
        # not necessarily label, only need its shape
        label, epsilon = inputs
        epsilon_tensor = tf.ones_like(label)
        return epsilon*epsilon_tensor
    
    epsilons = tf.vectorized_map(assign_epsilon, (label_dense, epsilon))
    outputs['epsilon'] = tf.reshape(epsilons, [-1,])

    return outputs
