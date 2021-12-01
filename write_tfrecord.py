import numpy as np 
import tensorflow as tf
import pandas as pd
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sklearn.preprocessing import LabelEncoder


is_train = False
if is_train:
    file_name = 'adult.data'
    write_dir = './data/train/block.tfrecord'
else:
    file_name = 'adult.test'
    write_dir = './data/eval/block.tfrecord'


training_data = pd.read_csv(file_name, sep=', ', header=None, dtype=str)

cols = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
print(training_data)
training_data.columns = cols

continuous_idx = [0, 2, 4, 10, 11, 12]

training_data = training_data.replace('?', np.nan)

training_data = training_data.dropna(axis=0, how='any')

data_df = pd.DataFrame(training_data)
le = LabelEncoder()

for i in range(len(cols)):
    if i in continuous_idx:
        data_df[cols[i]] = data_df[cols[i]].astype('float32')
    else:
        data_df[cols[i]] = le.fit_transform(data_df[cols[i]]).astype('float32')

data_np = data_df.values

def normalization(data):
    _range = np.max(data) - np.min(data)
    norm = (data - np.min(data)) / _range
    return norm

for i in range(data_np.shape[1]-1):
    data_np[:, i] = normalization(data_np[:, i])


print("Data_shape: ", data_np.shape)

if is_train:
    data_blocks = [data_np[1000*i:1000*(i+1), :] for i in range(30)]
else:
    data_blocks = [data_np]
# # print(data_blocks[0][:, 0])

def wrap_float32(val):
    return tf.train.Feature(float_list=tf.train.FloatList(value=val))

def wrap_int64(val):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=val))


def write_tfrecord(data, path, is_train):
    with tf.io.TFRecordWriter(path) as writer:
        if is_train:
            block_idx = 1
        else:
            block_idx = 0
        for i in range(len(data)):
            adult = {}
            adult['feature'] = wrap_float32(data[i][:, :-1].reshape(-1,))
            adult['label'] = wrap_float32(data[i][:, -1])
            adult['block'] = wrap_int64([block_idx])

            feature = tf.train.Features(feature=adult)
            example = tf.train.Example(features=feature)
            serialized = example.SerializeToString()
            writer.write(serialized)
            print("Block: ", block_idx)
            block_idx += 1

    return


write_tfrecord(data_blocks, write_dir, is_train)



