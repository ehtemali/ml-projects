from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load sample dataset
dataset = load_dataset('csv', data_files={'train':'../data/train.csv','test':'../data/test.csv'})

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)

# Tokenize
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)
dataset = dataset.map(tokenize, batched=True)

# Training
training_args = TrainingArguments(
    output_dir='./results', num_train_epochs=3, per_device_train_batch_size=8,
    evaluation_strategy='epoch', logging_dir='./logs'
)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset['train'], eval_dataset=dataset['test'])
trainer.train()
