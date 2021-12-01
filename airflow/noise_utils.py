
import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp

def laplace_noise_input(raw, sensitivity, epsilon):
    laplace = tfp.distributions.Laplace(0.0, (sensitivity/epsilon))
    def add_noise(clean):
        noise = laplace.sample(sample_shape=clean.shape)
        tf.print("Noise shape: ", noise.shape)
        noised = tf.math.add(noise, clean)
        
        return noised
    
    noised = tf.vectorized_map(add_noise, raw)

    return noised

def laplace_noise_gradient(gradient, sensitivity, epsilon, epoch):
    clipped_gradients = tf.clip_by_value(gradient, clip_value_min=-(sensitivity/2), clip_value_max=(sensitivity/2))
    eps_iter = epsilon/epoch
    laplace = tfp.distributions.Laplace(0.0, (sensitivity/eps_iter))
    noise = laplace.sample(sample_shape=gradient.shape)
    noised = tf.math.add(noise, gradient)
  
    return noised
