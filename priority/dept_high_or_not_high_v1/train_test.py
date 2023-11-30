# %% [markdown]
# # Importsand preparations

# %%
import pandas as pd
import torch
import os
import numpy as np
import datasets
import transformers
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from datasets import load_dataset, Dataset, DatasetDict

# %%
# !watch -n 0.5 nvidia-smi

# %%
print(f'PyTorch version: {torch.__version__}')  # 1.9.1+cu111
print(f'CUDA version: {torch.version.cuda}')  # 11.1
print(f'cuDNN version: {torch.backends.cudnn.version()}')  # 8005
print(f'Current device: {torch.cuda.current_device()}')  # 0
print(f'Is cuda available: {torch.cuda.is_available()}')  # TRUE

# %%
print(f'Transformers version: {transformers.__version__}')
print(f'Datasets version: {datasets.__version__}')

# %%
# Prevent a warning related to the tokenization process in the transformers library. 
os.environ["TOKENIZERS_PARALLELISM"] = "False"
# Makes CUDA operations synchronous
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# %%
# Find the GPU with the least memory usage.
!nvidia-smi

# %%
def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    # free unreferenced tensors from the GPU memory.
    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

free_gpu_cache()   

# %%
# Smaller and faster than bert.

model_ckpt = "bert-base-uncased"

epochs = 5 #Number of full cyles through the training set.
num_labels = 2 #Number of labels, high, med, low priority.
learning_rate = 5e-5 # Rate the model updates based on the data its trained on.
train_batch_size = 16 # Number of training examples in one iteration.
eval_batch_size = 32 # Number evalutaion examples in on iteratoion.
save_strategy = "no" # Should the model be saved automatically during training.
save_steps = 500 # How often to save the model during training. No effect since no over.
logging_steps = 100
model_dir = "./model" #Where to save model

# Use early stopping to prevent overfitting
#load_best_model_at_end=True
#metric_for_best_model="eval_loss"
#greater_is_better=False

# %% [markdown]
# Load dataset from huggingface

# %%
dataset = load_dataset("kristmh/high_priority_or_not_high_1")
dataset

# %%
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# %%
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)
#tokenizer = AutoTokenizer.from_pretrained(base_model_id)
# optim = torch.optim.Adam(model.parameters(), lr=5e-5)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
device

# %% [markdown]
# ## Tokenization

# %%
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# %% [markdown]
#     Tokenizing the whole dataset

# %%
#Tokenize the dataset to the correct input for the transformer model.
def tokenize(batch):
    return tokenizer(batch["text_clean"], padding="max_length", truncation=True)

# %%
tokenized_dataset = dataset.map(tokenize, batched=True)

# %%
train_dataset = tokenized_dataset["train"]
print(train_dataset)
validation_dataset = tokenized_dataset["validate"]
print(validation_dataset)
test_dataset = tokenized_dataset["test"]
test_dataset

# %% [markdown]
# ## Training a classifier

# %%
training_args = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    save_strategy=save_strategy,
    save_steps=save_steps,
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    logging_steps=logging_steps,
)

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)

# %%
trainer.train() 

# %% [markdown]
# * Training loss: Difference between the predictons made by the model on the training dataset vs on the actual data.
# * Validation loss: how well the model functions on unseen data.
# * Accuracy: How much the model gets correct. number of correct Prediction / total number of predictions.
# * F1: consider both precision and recall. 
# * Precision: Accuracy of positive predictions. Percison TP = TP + FP. How often the model is correct.
# * Recall: True positive rate. how many items the model gets correct from the total amount.

# %% [markdown]
# ### Training loss decreases, valdiation loss increases = Overfitting

# %%
# Evaluate validation set
eval_result = trainer.evaluate(eval_dataset=validation_dataset)

# %%
for key, value in sorted(eval_result.items()):
    print(f"{key} = {value}\n")

# %%
# Evaluate test data set
test_results = trainer.evaluate(eval_dataset=test_dataset)

# %%
for key, value in sorted(test_results.items()):
    print(f"{key} = {value}\n")

# %%
trainer.save_model(model_dir + "_local") 

# %%
from transformers import pipeline
    
classifier = pipeline("text-classification", model="./model_local")

# %%
classifier.model

# %%
classifier("this does not need to be done fast")

# %%
classifier("this is super important")

# %%
classifier("this bug has super high impact on the project")

# %% [markdown]
# ## Important to delete large objects to free memory 
# del train_dataset

# %%
del validation_dataset 

# %%
del model

# %%
# Free cache
torch.cuda.empty_cache()

# %%
!nvidia-smi



# %%
