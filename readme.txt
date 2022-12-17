We give the code of the example pipeline introduced in Appendix I here.

Preparation:
The Adult dataset could be downloaded from https://archive.ics.uci.edu/ml/datasets/adult
1. Make Directories. First, we need a 'data' folder containing (1) 'train' folder with training data named as 'block.tfrecord'; (2) 'eval' folder with testing data named as 'block.tfrecord'; (3) 'info' folder with two files, 'budget.info' stores the privacy budget status of data blocks, and 'allocation.info' stores the block selection and budget allocation for this pipeline; (3) 'filter' folder with the selected data blocks named as 'filter.tfrecord'. Second, we need a 'saved_model' folder saving the checkpoints of the model.
2. Write the dataset into TFRecords in forms of data blocks as we introduce in the paper, with each piece of TFRecord corresponding to a data block, containing three attributes: feature, label, and block_idx. (write_tfrecord.py) (Generating 'block.tfrecord' to /data/train/ and /data/eval/)
3. Write the current budget status, a numpy array in shape of (2, block_num), with its first row denoting the initial budget for blocks and the second row denoting the consumed budget for blocks. (write_info.py) (Generating 'budget.info' to /data/info/)
4. Save the current weights of the model to /saved_model/. (Or use adult_model.py to generate a randomized weight)


We implement DPlanner based on the Tensorflow Extended (TFX) and give two means for usage: by Jupyter notebook or Run in Airflow.

Jupyter Notebook:
It's an interactive way to run TFX. Open the adult.ipynb and run the code blocks.

Airflow:
It's an integrated way to run the pipeline. First put the 'adult_pipeline.py', 'noise_utils.py', and 'schedule_utils.py' in the dags directory of your airflow and change the file_path in 'adult_pipeline.py' to the path where you put the 'adult_trainer.py' and 'adult_transform.py'. Then open the airflow interface, refreshing the dags and run the adult pipeline.