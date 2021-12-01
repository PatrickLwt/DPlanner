
import tensorflow as tf 
import tensorflow_transform as tft

from tfx.components.trainer.fn_args_utils import FnArgs
from typing import List, Text
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options
import noise_utils
from adult_model import NN
from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision

noise_type = 0
_epoch = 100
_clip_value = 0.01

class NNNoised(NN):
    def __init__(self, clip_value, epoch):
        super(NNNoised, self).__init__()
        self.clip_value = clip_value
        self.epoch = epoch

    def add_noise(self, inputs):
        gradient, epsilon = inputs
        noised = tf.nest.map_structure(lambda g:noise_utils.laplace_noise_gradient(g, self.clip_value, epsilon, self.epoch), gradient)

        return noised

    def train_step(self, data):
        # feature_dict has 'epsilon' and 'feature_xf'
        feature_dict, labels = data
        with tf.GradientTape() as tape:
            predictions = self(feature_dict['feature_xf'], training=True)
            loss = self.compiled_loss(labels, predictions)
            regularization = sum(self.losses)
            loss += regularization

        jacobian = tape.jacobian(loss, self.trainable_variables)
        gradient = tf.vectorized_map(self.add_noise, (jacobian, feature_dict['epsilon']))
        gradient = tf.nest.map_structure(lambda g:tf.math.reduce_mean(g, axis=0), gradient)
        self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))

        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        feature_dict, labels = data
        predictions = self(feature_dict['feature_xf'], training=False)
        self.compiled_metrics.update_state(labels, predictions)

        return {m.name: m.result() for m in self.metrics}


class NNUnnoised(NN):
    def __init__(self, epoch):
        super(NNUnnoised, self).__init__()
        self.epoch = epoch
        
    def train_step(self, data):
        feature_dict, labels = data
        with tf.GradientTape() as tape:
            predictions = self(feature_dict['feature_xf'], training=True)
            loss = self.compiled_loss(labels, predictions, regularization_losses=self.losses)
        
        gradient = tape.gradient(loss, self.trainable_variables)            
        self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))
        self.compiled_metrics.update_state(labels, predictions)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        feature_dict, labels = data
        predictions = self(feature_dict['feature_xf'], training=False)
        self.compiled_metrics.update_state(labels, predictions)

        return {m.name: m.result() for m in self.metrics}

    
def build_keras_model(model_dir):
    if noise_type == 1:
        model = NNNoised(_clip_value, _epoch)
        loss_custom = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    else:
        model = NNUnnoised(_epoch)
        loss_custom = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.load_weights(model_dir)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, 215, 0.99, staircase=True)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=loss_custom, metrics=[AUC()], run_eagerly=False)
    
    return model


def input_fn(file_pattern: List[Text],
             data_accessor: DataAccessor,
             tf_transform_output: tft.TFTransformOutput,
             batch_size: int = 200) -> tf.data.Dataset:
    return data_accessor.tf_dataset_factory(
            file_pattern,
            dataset_options.TensorFlowDatasetOptions(
              batch_size=batch_size, label_key='label_xf'),
            tf_transform_output.transformed_metadata.schema).repeat()


def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
#         feature_spec.pop('label')
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        print("Transformed_features: ", transformed_features)
        return model(transformed_features['feature_xf'])

    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(fn_args.train_files, fn_args.data_accessor,
                                tf_transform_output, 250)
    eval_dataset = input_fn(fn_args.eval_files, fn_args.data_accessor,
                               tf_transform_output, 250)

    model = build_keras_model('/home/liweiting/adult/saved_model/')

    # Write logs to path
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')

    model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      epochs=model.epoch,
      verbose=2,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

    signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                  tf.TensorSpec(shape=[None], dtype=tf.string, name='transformed_examples'))
    }
    
    print("Saved_path: ", fn_args.serving_model_dir)
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
