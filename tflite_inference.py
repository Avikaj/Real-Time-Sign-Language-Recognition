import pandas as pd
import numpy as np
import tensorflow.lite as tflite
import matplotlib.pyplot as plt

ROWS_PER_FRAME = 543 

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

interpreter = tflite.Interpreter("./model.tflite")
found_signatures = list(interpreter.get_signature_list().keys())
prediction_fn = interpreter.get_signature_runner("serving_default")

train = pd.read_csv('train.csv')
train['sign_ord'] = train['sign'].astype('category').cat.codes

SIGN2ORD = train[['sign', 'sign_ord']].drop_duplicates().set_index('sign').squeeze().to_dict()
ORD2SIGN = train[['sign_ord', 'sign']].drop_duplicates().set_index('sign_ord').squeeze().to_dict()

pq_file = 'output.parquet'
xyz_np = load_relevant_data_subset(pq_file)
prediction = prediction_fn(inputs = xyz_np)
sign = prediction['outputs'].argmax()
