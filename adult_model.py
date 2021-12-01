import tensorflow as tf 
from tensorflow.keras.layers import Dense, Softmax, BatchNormalization
from tensorflow.keras import Model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import noise_utils

# _noise_type = 1
# _clip_value = 0.01
# _epoch = 40

class NN(Model):
    def __init__(self):
        super(NN, self).__init__()
        self.d1 = Dense(8, activation='sigmoid', name='fc1')
        self.bn = BatchNormalization()
        self.d2 = Dense(1, activation='sigmoid', name='fc2')

    def call(self, x):
        x = self.d1(x)
        x = self.bn(x)
        x = self.d2(x)
        return x
    
#     def add_noise(self, inputs):
#         gradient, epsilon = inputs
#         noised = tf.nest.map_structure(lambda g:noise_utils.laplace_noise_gradient(g, self.clip_value, epsilon, self.epoch), gradient)
        
#         return noised
        
#     def train_step(self, data):
#         # feature_dict has 'epsilon' and 'feature_xf'
#         feature_dict, labels = data
#         with tf.GradientTape() as tape:
#             predictions = self(feature_dict['feature_xf'], training=True)
#             loss = self.compiled_loss(labels, predictions)
#             regularization = sum(self.losses)
#             loss += regularization
  
#         if self.noise_type == 1:
#             jacobian = tape.jacobian(loss, self.trainable_variables)
#             gradient = tf.vectorized_map(self.add_noise, (jacobian, feature_dict['epsilon']))
#             gradient = tf.nest.map_structure(lambda g:tf.math.reduce_mean(g, axis=0), gradient)
#             self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))
#         else:
#             gradient = tape.gradient(loss, self.trainable_variables)            
#             self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))

#         self.compiled_metrics.update_state(labels, predictions)
#         return {m.name: m.result() for m in self.metrics}

#     def test_step(self, data):
#         feature_dict, labels = data
#         predictions = self(feature_dict['feature_xf'], training=False)
#         self.compiled_metrics.update_state(labels, predictions)

#         return {m.name: m.result() for m in self.metrics}

    
if __name__ == "__main__":
    # Initialized a random model 
    model = NN()
    inputs = tf.random.uniform((1, 14))
    outputs = model(inputs)
    model.summary()
    for layer in model.layers:
        print(layer.get_weights())
    model.save_weights('/home/liweiting/adult/saved_model/weights.h5', save_format='h5')
    print("Model Saved!")
    
