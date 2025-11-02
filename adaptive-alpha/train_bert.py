import os
import json
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback

# Define the callback
class SaveMetricsCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self.log_history.append(logs)
            with open(self.log_path, 'w') as f:
                json.dump(self.log_history, f, indent=4)

def load_data_from_directory(directory_path):
    """Loads data from a directory of JSON files into a pandas DataFrame."""
    records = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                if 'error' not in data and 'image_dependence' in data:
                    records.append({
                        'text': data.get('text'),
                        'label': float(data.get('image_dependence'))
                    })
    return pd.DataFrame(records)

class RegressionDataset(torch.utils.data.Dataset):
    """Custom PyTorch Dataset for our regression task."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    """Computes Mean Squared Error for evaluation."""
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    return {'mse': mse}

def main():
    # 1. Detect device
    print("Detecting available device...")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS (Apple Silicon GPU) is available.")
    else:
        print("MPS not available. Training will use the CPU (this may be slow).")

    # 2. Load and prepare data
    print("Loading and preparing data...")
    df = load_data_from_directory('adaptive-alpha/analysis_results')
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

    # 3. Initialize tokenizer and tokenize the data
    print("Tokenizing data...")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_texts = train_df['text'].tolist()
    eval_texts = eval_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    eval_labels = eval_df['label'].tolist()

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=512)

    train_dataset = RegressionDataset(train_encodings, train_labels)
    eval_dataset = RegressionDataset(eval_encodings, eval_labels)

    # 4. Load the model
    print("Loading BERT model for regression...")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # 5. Define Training Arguments using older, more compatible parameters
    # Calculate steps per epoch to save the model at the end of each epoch
    steps_per_epoch = len(train_dataset) // 8 # 8 is per_device_train_batch_size
    if steps_per_epoch == 0: steps_per_epoch = 1

    training_args = TrainingArguments(
        output_dir='adaptive-alpha/bert_regression_trainer',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="steps",       # ✅ 旧版本参数名
        save_strategy="steps",
        save_steps=steps_per_epoch,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False
    )

    # Instantiate the callback
    save_metrics_callback = SaveMetricsCallback(log_path="training_log.json")

    # 6. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[save_metrics_callback]
    )

    # 7. Train the model
    print("Starting training...")
    trainer.train()

    # 8. Save the final model and tokenizer
    print("Training finished. Saving the best model...")
    trainer.save_model("adaptive-alpha/bert_regression_final")
    tokenizer.save_pretrained("adaptive-alpha/bert_regression_final")
    print("Model saved to ./bert_regression_final")

if __name__ == "__main__":
    main()