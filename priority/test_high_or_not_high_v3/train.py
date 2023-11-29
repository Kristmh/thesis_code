from tabnanny import check
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, get_scheduler
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.optim import AdamW
from tqdm.auto import tqdm

data = pd.read_csv("csv/clean_priority_high_or_not_high.csv" , index_col = 0)

# Split dataframe into three parts: training, validation and testing.
def train_validate_test_split(df, train_percent=.8, validate_percent=.1, seed=42):
    np.random.seed(seed)
    # Shuffle index of dataframe
    perm = np.random.permutation(df.index)
    
    df_length = len(df.index)
    
    # Number of row in training set
    train_end = int(train_percent * df_length)
    # Number of rows in validate set
    validate_end = int(validate_percent * df_length) + train_end
    
    # From start to train end
    train = df.iloc[perm[:train_end]]
    # From train_end to validate_end
    validate = df.iloc[perm[train_end:validate_end]]
    # From validate to the last row in dataframe.
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

# 80% training, 10% validate, 10% test. Seed 42.
# Test 80-10-10 and 70-15-15
train , validate , test = train_validate_test_split(data)

train.set_index("label" , inplace = True)
validate.set_index("label" , inplace = True)
test.set_index("label" , inplace = True)

# Convert from Pandas DataFrame to Hugging Face datasets
tds = Dataset.from_pandas(train)
vds = Dataset.from_pandas(validate)
testds = Dataset.from_pandas(test)

ds = DatasetDict()

ds["test"] = testds
ds["train"] = tds
ds["validate"] = vds



accelerator = Accelerator()

checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

def tokenize_function(example):
    return tokenizer(example["text_clean"], truncation=True)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
optimizer = AdamW(model.parameters(), lr=3e-5)


# Tokenize all the dataset
tokenized_datasets = ds.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Remove unnecessary columns that the model does not expect.
tokenized_datasets = tokenized_datasets.remove_columns(["text_clean"])
# Rename the column to labels because the model expect that.
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# Returns PyTorch tensors instead of lists.
tokenized_datasets.set_format("torch")


train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validate"], batch_size=8, collate_fn=data_collator
)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)