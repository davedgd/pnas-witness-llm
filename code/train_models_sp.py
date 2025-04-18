import torch
import numpy as np
import pandas as pd
import janitor

import os
os.environ['WANDB_SILENT'] = 'true'

from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset

import warnings
warnings.filterwarnings('ignore', message = 'Importing display from IPython.core.display is deprecated since IPython 7.14.*')
warnings.filterwarnings('ignore', message = """
    There is an imbalance between your GPUs.*""")
warnings.filterwarnings('ignore', message = 'Was asked to gather along dimension 0, but all input tensors were scalars.*')

#from transformers import logging
#logging.set_verbosity_error()

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TextClassificationPipeline
)

import matplotlib.pyplot as plt

from scipy.special import softmax
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_recall_curve, balanced_accuracy_score, accuracy_score, ConfusionMatrixDisplay, classification_report
from netcal.metrics import ECE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import random
from transformers import set_seed
seed = 123

set_seed(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

import sys

pretrained_model_name_or_path = sys.argv[1]
print('*** Begin Fine Tuning:', sys.argv[1], '***')

data = pd.read_csv('../../data/processed/train.csv')
data['target'] = data['target'].astype('str').replace({'0': 'Inaccurate', '1': 'Accurate'})
data['text_a'] = data['expression_spellcheck'].astype('str')
data['text_b'] = data['justification_spellcheck'].astype('str')
data = data[['text_a', 'text_b', 'target']]

train = data.sample(frac = 0.80, random_state = seed)
val = data[~data.index.isin(train.index)]

test = pd.read_csv('../../data/processed/test.csv')
test['target'] = test['target'].astype('str').replace({'0': 'Inaccurate', '1': 'Accurate'})
test['text_a'] = test['expression_spellcheck'].astype('str')
test['text_b'] = test['justification_spellcheck'].astype('str')
test = test[['text_a', 'text_b', 'target']]

label2id = {
    'Inaccurate': 0,
    'Accurate': 1
    }

id2label = {
    0: 'Inaccurate',
    1: 'Accurate'
    }

ds_train = Dataset.from_dict({'text_a': train.text_a, 
                              'text_b': train.text_b, 
                              'labels': train.target})

ds_val = Dataset.from_dict({'text_a': val.text_a, 
                            'text_b': val.text_b, 
                            'labels': val.target})

ds_test = Dataset.from_dict({'text_a': test.text_a, 
                             'text_b': test.text_b, 
                             'labels': test.target})

dataset_dict = DatasetDict({
    'train': ds_train, 
    'val': ds_val, 
    'test': ds_test
})

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

def encode (examples):
    tokenized_examples = tokenizer(examples['text_a'], examples['text_b'], return_token_type_ids = True)
    tokenized_examples['labels'] = [label2id[label] for label in examples['labels']]
    return tokenized_examples

dataset_dict_tokenized = dataset_dict.map(
    encode,
    batched = True,
    num_proc = os.cpu_count(),
    remove_columns = ['text_a', 'text_b']
)

data_collator = DataCollatorWithPadding(tokenizer, padding = True)

if "ModernBERT" in pretrained_model_name_or_path:
    def model_init ():
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            num_labels = len(test.groupby('target').size()),
            id2label = id2label,
            label2id = label2id,
            torch_dtype = torch.bfloat16
            )
else:
    def model_init ():
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            num_labels = len(test.groupby('target').size()),
            id2label = id2label,
            label2id = label2id,
            )
    
def compute_metrics (eval_pred):
    predictions, labels = eval_pred
    preds = [int(np.argmax(prediction)) for prediction in predictions]
    probs = softmax(predictions, axis = -1)[:, 1]
    return {'eval_auc': roc_auc_score(labels, probs),
            'eval_bal_acc': balanced_accuracy_score(labels, preds)}

def eval_bal_acc (metrics):
    return metrics['eval_bal_acc']

def eval_auc (metrics):
    return metrics['eval_auc']

project = sys.argv[2]
goal = 'maximize'

def wandb_hp_space (trial):
    return {
        'project': project,
        'method': 'grid',
        'metric': {'name': 'objective', 'goal': goal},
        'parameters': {
            'learning_rate': {'values': [5.0e-06, 8.0e-06, 9.0e-06, 1.0e-05, 1.5e-05, 2.0e-05, 2.5e-05,
       3.0e-05, 4.0e-05, 5.0e-05]},
            'per_device_train_batch_size': {'values': [8, 16, 32]},
            'num_train_epochs': {'values': [1, 2, 3, 4, 5, 10]}
        },
    }

args = TrainingArguments(
    'training_output',
    per_device_eval_batch_size = 32,
    eval_strategy = 'epoch',
    save_strategy = 'no'
)

trainer = Trainer(
    args = args,
    data_collator = data_collator,
    model_init = model_init,
    train_dataset = dataset_dict_tokenized['train'],
    eval_dataset = dataset_dict_tokenized['val'],
    compute_metrics = compute_metrics
)

best_trial = trainer.hyperparameter_search(
    direction = goal,
    backend = 'wandb',
    hp_space = wandb_hp_space,
    n_trials = None,
    compute_objective = eval_bal_acc
)

best_settings_df = pd.DataFrame({
    'model': pretrained_model_name_or_path,
    'learning_rate': best_trial.hyperparameters['learning_rate'],
    'per_device_train_batch_size': best_trial.hyperparameters['per_device_train_batch_size'],
    'num_train_epochs': best_trial.hyperparameters['num_train_epochs']
}, index = [0])
best_settings_df.to_csv('../sp/results/' + pretrained_model_name_or_path.replace('/', '_') + '_hyperparams.csv', index = False)

# refit model via True; otherwise, False
if False:
    best_trial = lambda: None
    best_trial.hyperparameters = {
        'learning_rate': 5e-05,
        'per_device_train_batch_size': 32,
        'num_train_epochs': 3
    }

final_args = TrainingArguments(
    'training_output',
    learning_rate = best_trial.hyperparameters['learning_rate'],
    per_device_train_batch_size = best_trial.hyperparameters['per_device_train_batch_size'],
    num_train_epochs = best_trial.hyperparameters['num_train_epochs'],
    per_device_eval_batch_size = 32,
    eval_strategy = 'epoch',
    save_strategy = 'epoch',
    load_best_model_at_end = True,
    #metric_for_best_model = 'eval_bal_acc',
    #greater_is_better = True,
    report_to = 'none'
)

final_trainer = Trainer(
    args = final_args,
    data_collator = data_collator,
    model_init = model_init,
    train_dataset = dataset_dict_tokenized['train'],
    eval_dataset = dataset_dict_tokenized['val'],
    compute_metrics = compute_metrics
)

model_checkpoint = '../sp/models/' + pretrained_model_name_or_path.replace('/', '_')
final_trainer.train()
final_trainer.save_model(model_checkpoint)
tokenizer.save_pretrained(model_checkpoint)

data_test_dict = []

for i, _ in test.iterrows():
    data_test_dict.append({'text': test['text_a'][i], 
                           'text_pair': test['text_b'][i]})

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels = len(test.groupby('target').size())
).to(device)

pipe = TextClassificationPipeline(model = model, tokenizer = tokenizer, top_k = None, device = device)

# temporary workaround for XLNet batch size issue
if str(model.base_model).find('XLNetModel') != -1:
    batch_size = 1
else:
    batch_size = 128

raw_probs = pipe(data_test_dict, batch_size = batch_size)

probs = np.array([[i['score'] for i in item if i['label'] == id2label[1]][0] for item in raw_probs])
preds = np.where(probs >= 0.5, 1, 0)

y_true = [label2id[x] for x in test['target']]

out = pd.DataFrame({
    'text_a': test['text_a'],
    'text_b': test['text_b'],
    'target': test['target'],
    'pred': [id2label[x] for x in preds],
    'prob': probs
})

out.to_csv('../sp/results/' + pretrained_model_name_or_path.replace('/', '_') + '_preds.csv', index = False)

n_bins = 7
ece = np.round(ECE(bins = n_bins).measure(np.array(probs), np.array(y_true)), 3)

print('AUC:', np.round(roc_auc_score(y_true, probs), 3))
print('Balanced Accuracy:', np.round(balanced_accuracy_score(y_true, preds), 3))
print('Macro F1:', np.round(f1_score(y_true, preds, average = 'macro'), 3))
print('ECE', ece)
print('*** End Fine Tuning:', sys.argv[1], '***')