import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pprint
import tempfile
import urllib
import datetime
import random
from typing import List

import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma
tf.get_logger().propagate = False
import numpy as np 

from tfx.proto import example_gen_pb2
from tfx.orchestration import pipeline, metadata
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner, AirflowPipelineConfig
from tfx.v1.proto import TrainArgs, EvalArgs, Input, PushDestination
from tfx.components import ImportExampleGen, StatisticsGen, SchemaGen, Transform, Trainer, Evaluator, Pusher

from tfx.dsl.component.experimental.decorators import component
from tfx.v1.dsl.components import InputArtifact, OutputArtifact, Parameter, OutputDict
from tfx.types.standard_artifacts import Examples, Model, String
from tfx.types import standard_component_specs
from tfx.utils import proto_utils
from tfx.components.example_gen.import_example_gen import executor

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout, Activation
from tensorflow.keras import Model, regularizers, Sequential

import schedule_utils

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


_pipeline_name = 'adult_custom'
_adult_root = os.path.join(os.environ['HOME'], 'adult')
_data_root = os.path.join(os.environ['HOME'], 'adult', 'data')
_serving_model_dir = os.path.join(os.environ['HOME'], 'adult', 'saved_model', 'weights.h5')   ## path to save the trained model
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)

_info_root = os.path.join(_data_root, 'info')  # path to save the budget info and epsilon allocation info
_filter_root = os.path.join(_data_root, 'filter')  # path to save the chosen blocks tfrecord

_transform_module_file = os.path.join(os.environ['HOME'], 'adult', 'adult_transform.py')
_trainer_module_file = os.path.join(os.environ['HOME'], 'adult', 'adult_trainer.py')

_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]

_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2019, 1, 1),
}

feature_num = 14
@component
def Estimator(data:InputArtifact[Examples],
              significance: OutputArtifact[String],
              model_dir: Parameter[str]):
    feature_description = {}
    feature_description['feature'] = tf.io.VarLenFeature(tf.float32)
    feature_description['label'] = tf.io.VarLenFeature(tf.float32)
    feature_description['block'] = tf.io.VarLenFeature(tf.int64)
    
    tags = ['feature', 'label', 'block']
    
    def read_and_decode(example_string):
        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        outputs = [feature_dict[tags[i]] for i in range(len(tags))]
        return outputs
    
    train_uri = os.path.join(data.uri, 'Split-train')
    train_filenames = [os.path.join(train_uri, name) for name in os.listdir(train_uri)]
    trainset = tf.data.TFRecordDataset(train_filenames, compression_type='GZIP')
    blocks = trainset.map(read_and_decode)
    
    val_uri = os.path.join(data.uri, 'Split-eval')
    val_filenames = [os.path.join(val_uri, name) for name in os.listdir(val_uri)]
    valset = tf.data.TFRecordDataset(val_filenames, compression_type='GZIP')
    val_data = valset.map(read_and_decode) 
    
    # Only one item in val_data set and here aiming to extract this item
    for val in val_data:
        val_feature, val_label, _ = val
        val_feature = tf.reshape(tf.sparse.to_dense(val_feature), [-1, feature_num])
        val_label = tf.reshape(tf.sparse.to_dense(val_label), [-1,])
    
    def compile_model(model):
        loss_custom = tf.keras.losses.BinaryCrossentropy()
#     lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, 215, 0.99, staircase=True)
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss=loss_custom, metrics=[], run_eagerly=False)
        
        return model
    
    model = NN()
    inputs = tf.random.uniform((1, 14))
    outputs = model(inputs)
    model.load_weights(model_dir)
    model = compile_model(model)
    
    loss_before = model.evaluate(x=val_feature, y=val_label, batch_size=256)
    significance_list = []
    
    for block in blocks:    
        # Deal with the data
        train_feature, train_label, block_idx = block
        train_feature = tf.reshape(tf.sparse.to_dense(train_feature), [-1, feature_num])
        train_label = tf.reshape(tf.sparse.to_dense(train_label), [-1,])
        # Load the current model
        model.load_weights(model_dir)
        model = compile_model(model)
        
        hist = model.fit(x=train_feature, y=train_label, batch_size=250, epochs=10, verbose=0,
                         validation_data=(val_feature, val_label), validation_batch_size=256)
        loss_after = hist.history['val_loss'][-1]
        
        delta_loss = loss_before - loss_after
        significance_list.append(delta_loss)
        
    def normalization(data):
        _range = np.max(data) - np.min(data)
        norm = (data - np.min(data)) / _range
        return norm
    
    # Normalize the significance to [0, 1]
    significance_list = normalization(significance_list)
    print("Sig_list: ", significance_list)
    # significance_list = [1.0, 0.9, 0.9, 0.8, 1.0, 0.9, 0.9, 0.8, 1.0, 0.9, 0.9, 0.8, 1.0, 0.9, 0.9, 0.8,
    #                      1.0, 0.9, 0.9, 0.8, 1.0, 0.9, 0.9, 0.8,1.0, 0.9, 0.9]
    
    with tf.io.gfile.GFile(significance.uri, 'w') as f:
        for sig in significance_list:
            f.write(str(sig))
            f.write('\n')
    
    return

def sparse_to_dense(value):
    if isinstance(value, tf.sparse.SparseTensor):
        return tf.sparse.to_dense(value)
    return value

def wrap_int64(val):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=val))

def wrap_float32(val):
    return tf.train.Feature(float_list=tf.train.FloatList(value=val))


@component
def Scheduler(data: InputArtifact[Examples],
              significance: InputArtifact[String],
              blocks: OutputArtifact[Examples],
              data_root: Parameter[str],
              info_root: Parameter[str],
              schedule_type: Parameter[int],  # 0 for sage; 1 for infocom; 2 for rpbs; 3 for extended rpbs
              gamma: Parameter[float],    # gamma is the neighborhood error range
              lambd: Parameter[float]):   # lambd is the probability lower bound, presetted when the pipeline is created
    train_uri = os.path.join(data.uri, 'Split-train')
    train_filenames = [os.path.join(train_uri, name) for name in os.listdir(train_uri)]
    dataset = tf.data.TFRecordDataset(train_filenames, compression_type='GZIP')

    feature_description = {}
    feature_description['feature'] = tf.io.VarLenFeature(tf.float32)
    feature_description['label'] = tf.io.VarLenFeature(tf.float32)
    feature_description['block'] = tf.io.VarLenFeature(tf.int64)
    
    tags = ['feature', 'label', 'block']
    
    def read_and_decode(example_string):
        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        outputs = [feature_dict[tags[i]] for i in range(len(tags))]
        return outputs
    
    records = dataset.map(read_and_decode)
    
    sig_list = []
    sig_uri = significance.uri
    f = open(sig_uri, 'r')
    for sig in f.readlines():
        sig_list.append(float(sig))
    f.close()
    print("Sig_list: ", sig_list)
    
    info_dir = os.path.join(info_root, 'budget.info')
    eps_dir = os.path.join(info_root, 'allocation.info')
    block_dir = os.path.join(data_root, 'filter/filter.tfrecord')
    
    sensitivity = 1.0 * feature_num
    
    info = np.loadtxt(info_dir) ## default delimiter is " "
    eps_array = np.array([[0.], [0.001]])
    block_list = list(range(1, (info.shape[1]+1)))
    impl_Flag = True
    eps0 = 1.0
    # Randomly choose a block
    while(impl_Flag):
        idx = random.choice(block_list)
        block_list.remove(idx)
        significance = sig_list[(idx-1)]
        print("Significacne: ", significance)
        initial_budget = info[0, idx-1]
        consumed_budget = info[1, idx-1]
        
        # Block Retire
        if consumed_budget > initial_budget:
            continue
        
        args = [initial_budget, consumed_budget, lambd, gamma, sensitivity, significance, eps0]
        
        # Allocation Strategy            
        epsilon = schedule_utils.schedule(schedule_type=schedule_type, arguments=args)
        
        if epsilon>0:
            info[1, idx-1] += (epsilon+eps0)
            eps_array = np.concatenate((eps_array, np.array([[idx], [epsilon]])), axis=1)
            print("Epislon: ", epsilon)
            
        # Stop Rule
        if eps_array.shape[1] > 5 or len(block_list)==0:
            impl_Flag = False
    
            
    # Update the info file
    np.savetxt(info_dir, info, fmt='%f')
    
    # Save the allocation result
    np.savetxt(eps_dir, eps_array, fmt='%f')
    print('Eps array: ', eps_array)
    
    # Package the selected blocks as an artifact
    def package(record):
        content = {}
        content['feature'] = wrap_float32(sparse_to_dense(record[0]).numpy())
        content['label'] = wrap_float32(sparse_to_dense(record[1]).numpy())
        content['block'] = wrap_int64(sparse_to_dense(record[2]).numpy())
        feature = tf.train.Features(feature=content)
        example = tf.train.Example(features=feature)
        serialized = example.SerializeToString()
        
        return serialized
    
    allocation_list = np.delete(eps_array[0, :], 0)
    print("Allocation: ", allocation_list)
    
    with tf.io.TFRecordWriter(block_dir) as writer:
        for record in records:
            idx = sparse_to_dense(record[-1])
            print("IDX: ", idx)
            if idx in allocation_list:
                print("idx: ", idx)
                writer.write(package(record))
    
    filter_input_cfg = Input(splits=[example_gen_pb2.Input.Split(name='train', pattern='filter/*'),
                                    example_gen_pb2.Input.Split(name='eval', pattern='eval/*')])
    
    filter_output_cfg = example_gen_pb2.Output()
    
    output_dict = {standard_component_specs.EXAMPLES_KEY: [blocks]}
    exec_properties = {standard_component_specs.INPUT_CONFIG_KEY: proto_utils.proto_to_json(filter_input_cfg),
                      standard_component_specs.OUTPUT_CONFIG_KEY: proto_utils.proto_to_json(filter_output_cfg),
                      standard_component_specs.INPUT_BASE_KEY: data_root,
                      standard_component_specs.OUTPUT_DATA_FORMAT_KEY: example_gen_pb2.FORMAT_TF_EXAMPLE}
    example_gen = executor.Executor()
    example_gen.Do({}, output_dict, exec_properties)

    return


def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str, 
                     transform_module_file: str, trainer_module_file: str,
                     serving_model_dir: str, info_root: str, filter_root: str,
                     metadata_path: str, beam_pipeline_args: List[str]) -> pipeline.Pipeline:
    # Component 1: Import data from the database
    example_input_cfg = Input(splits=[example_gen_pb2.Input.Split(name='train', pattern='train/*'),
                                      example_gen_pb2.Input.Split(name='eval', pattern='eval/*')])
    example_gen = ImportExampleGen(input_base=data_root, input_config=example_input_cfg).with_id("Example_Gen")

    # Component 2: Calculate the significance
    estimator = Estimator(data=example_gen.outputs['examples'], model_dir=serving_model_dir)

    # Component 3: Scheduler to choose blocks and allocate privacy budget
    scheduler = Scheduler(data=example_gen.outputs['examples'], significance=estimator.outputs['significance'],
                          data_root=data_root, info_root=info_root, schedule_type=2, gamma=10.0, lambd=0.85)

    # Component 4: StatisticsGen, show the features of filter_gen
    statistics_gen = StatisticsGen(examples=scheduler.outputs['blocks'])

    # Component 5: SchemaGen, get the schema of filter_gen
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'], infer_feature_shape=False)

    # Component 6: Add noise to the clean data
    eps_dir = os.path.join(info_root, 'allocation.info')
    transform = Transform(examples=scheduler.outputs['blocks'], schema=schema_gen.outputs['schema'],
                          module_file=transform_module_file, custom_config=eps_dir)

    # Component 7: Train the Query's model
    trainer = Trainer(module_file=trainer_module_file,
                      examples=transform.outputs['transformed_examples'],
                      transform_graph=transform.outputs['transform_graph'],
                      schema=schema_gen.outputs['schema'],
                      train_args=TrainArgs(num_steps=30),
                      eval_args=EvalArgs(num_steps=16))

    # Component 8: Evaluate the performance of the model
    eval_config = tfma.EvalConfig(model_specs=[tfma.ModelSpec(label_key='label')],
                                  slicing_specs=[tfma.SlicingSpec()],
                                  metrics_specs=[tfma.MetricsSpec(metrics=[tfma.MetricConfig(
                                                 class_name='CategoricalAccuracy',
                                                 threshold=tfma.MetricThreshold(
                                                 value_threshold=tfma.GenericValueThreshold(
                                                 lower_bound={'value': 0.6})))])])
    evaluator = Evaluator(examples=scheduler.outputs['blocks'],
                          model=trainer.outputs['model'],
                          eval_config=eval_config)

    # Component 9: Release the model
    pusher = Pusher(model=trainer.outputs['model'],
                    model_blessing=evaluator.outputs['blessing'],
                    push_destination=PushDestination(filesystem=PushDestination.Filesystem(
                                                     base_directory=serving_model_dir)))

    return pipeline.Pipeline(pipeline_name=pipeline_name, pipeline_root=pipeline_root,
                             components=[example_gen, estimator, scheduler, statistics_gen,
                                         schema_gen, transform, trainer, evaluator, pusher,],
                             enable_cache=True, 
                             metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
                             beam_pipeline_args=beam_pipeline_args)



absl.logging.set_verbosity(absl.logging.INFO)
DAG = AirflowDagRunner(AirflowPipelineConfig(_airflow_config)).run(
     _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          data_root=_data_root,
          transform_module_file=_transform_module_file,
          trainer_module_file=_trainer_module_file,
          serving_model_dir=_serving_model_dir,
          info_root=_info_root,
          filter_root=_filter_root,
          metadata_path=_metadata_path,
          beam_pipeline_args=_beam_pipeline_args))

