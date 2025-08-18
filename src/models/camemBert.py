
# intended to run with a T4 GPU, update batch size accordingly

# #import files from gogoel drive if you want to run this code in Google Colab make sure to update all the paths accordingly
# from google.colab import drive
# drive.mount('/content/drive')


import pandas as pd
import numpy as np

import os
from datasets import Dataset
# check if data/processed exists, if not create it
if not os.path.exists('data/processed'):
    os.makedirs('data/processed')
    # Load the dataset, I selected only 5000 sample because of memory limitation
    df = pd.read_csv('data/language_analysis/df_localization.csv')
    #create a text column with translations where necessary
    df["text"] = np.where(df["deepL_translation"].notna(), df["deepL_translation"],
                                np.where(df["lang"] == "fr", df["merged_text"], np.nan))
    #drop superfluous lines
    df = df[['prdtypecode','text']]
    original_length = df.shape[0]
    df = df.dropna().reset_index(drop=True)
    cropped_length = df.shape[0]
    print(f'dropped {original_length - cropped_length:,} lines due to missing translations')

    # Encode all labels BEFORE splitting
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    df['lables'] = le.fit_transform(df['prdtypecode'])
    df.drop('prdtypecode', axis = 1, inplace = True)
    # we want to split test and training set
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['lables'], random_state=42)

    # !pip install datasets #needed for running the code in Google Colab
    # Convert to Hugging Face Datasets

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_dataset.to_csv('data/processed/camembert_train.csv')
    test_dataset.to_csv('data/processed/camembert_test.csv')

else :
    if os.path.exists('data/processed/camembert_train.csv'):
        train_dataset = pd.read_csv('data/processed/camembert_train.csv')
        test_dataset = pd.read_csv('data/processed/camembert_test.csv')  
    else:
        train_df = pd.read_csv('data/processed/train.csv')
        test_df = pd.read_csv('data/processed/test.csv')

        le = LabelEncoder()
        train_df['lables'] = le.fit_transform(train_df['prdtypecode'])
        test_df['lables'] = le.transform(test_df['prdtypecode'])
        train_df.drop('prdtypecode', axis = 1, inplace = True)
        test_df.drop('prdtypecode', axis = 1, inplace = True)
        # we want to split test and training set
       
        train_df = pd.to_csv('data/processed/camembert_train.csv')
        test_df = pd.to_csv('data/processed/camembert_test.csv')    

output_dir = '/models/camembert/class_weights_checkpoints/epoch'
logging_dir = '/models/camembert/class_weights_checkpoints/logs'

from transformers import AutoModelForSequenceClassification, AutoTokenizer, CamembertTokenizer
model_path = "/models/bert"
tokenizer_path = "/models/bert"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

import pickle
import torch
with open('/models/camembert/class_weights_f1_checkpoints/model_tokenizer.pkl', 'wb') as f:
    pickle.dump({'model': model, 'tokenizer': tokenizer}, f)

tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base-wikipedia-4gb", do_lower_case=True)
# Define the maximum length for the model (514 for Camembert because of the special tokens)
max_length = 256  #changed to 512 instead of 514, according to Camembert documentation and testing

def tokenize_function(examples):
    # Tokenize and ensure padding/truncation to max_length
    outputs = tokenizer(
        examples["text"],
        padding="max_length",         # Pad to max_length (ensures consistent length)
        truncation=True,               # Truncate sequences longer than max_length
        max_length=max_length,        # Ensure each sequence is exactly of max_length
        return_tensors="pt"           # Return PyTorch tensors
    )
    # Include labels in the output
    outputs["labels"] = examples["lables"]
    return outputs

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set the format for the dataset
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

tokenizer.save_pretrained("/models/camembert/class_weights_checkpoints/tokenizer")

import torch
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_labels = np.unique(train_df['lables'])
weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=train_df['lables'])
class_weights = torch.tensor(weights, dtype=torch.float)

print("Class Weights:", class_weights)



# ! pip install evaluate # needed for running the code in Google Colab
from transformers import CamembertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from torch.nn import CrossEntropyLoss
from evaluate import load
from sklearn.metrics import accuracy_score, precision_recall_fscore_support # import here

class PrintLossCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        print(f"Epoch {state.epoch:.0f} completed.")
        # Check if log_history is not empty before accessing elements
        if state.log_history:
            print(f"Training Loss: {state.log_history[-1].get('loss')}")
            print(f"Evaluation Loss: {state.log_history[-1].get('eval_loss')}")
            print(f"F1: {state.log_history[-1].get('eval_f1')}")
        else:
            print("Training and evaluation losses not available yet.")

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # add num_items_in_batch=None
        # Get labels from model output instead of input dictionary
        outputs = model(**inputs)  # Get model outputs (includes logits and hidden states)
        logits = outputs.logits
        labels = inputs.get("labels")  # Get labels from inputs if available. This will handle missing labels

        if labels is None:
          # Handle the case where labels are missing if necessary.
          # This could involve logging a warning, using a default value,
          #  or modifying your data loading process to include labels.
          # For now, we'll print a message and return a dummy loss:
          print("Warning: Labels not found in inputs.")
          return torch.tensor(0.0, device=logits.device)

        loss_fct = CrossEntropyLoss(weight=class_weights.to(logits.device))  # Apply class weights
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))  # Reshape if needed
        return (loss, outputs) if return_outputs else loss

    def load_model(self, *args, **kwargs):
        # Call the original load_model method from the Trainer class
        super().load_model(*args, **kwargs)

# Load the accuracy metricfrom evaluate import load
metric = load("accuracy")

# Initialize model and training arguments
model = CamembertForSequenceClassification.from_pretrained("camembert/camembert-base-wikipedia-4gb", num_labels=len(class_labels))

# Define compute_metrics outside the CustomTrainer class
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
    'accuracy': acc,
    'f1': f1,
    'precision': precision,
    'recall': recall
    }

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    # Use 'f1' as the metric for best model selection
    metric_for_best_model="f1",
    greater_is_better=True,  # F1 score is better when higher
    per_device_train_batch_size=124,
    per_device_eval_batch_size=248,
    remove_unused_columns=False, # Add this line
    num_train_epochs=10,
    logging_dir=logging_dir,
    logging_steps=0.1,
    report_to="none"
)

# Create and train the trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    callbacks=[PrintLossCallback()],
    compute_metrics=compute_metrics  # Calculate F1 score during evaluation
)

trainer.train()
# run trainer.train(resume_from_checkpoint=True) if you want to continue training from the last checkpoint

model.eval()  # Set to eval mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Tokenize and create batches
X_test = test_df.text.tolist()  # Convert Series to list if needed
y_test = test_df.lables

# Store predictions here
all_logits = []

# Process test data in batches
for i in range(0, len(X_test), batch_size):
    batch_texts = X_test[i:i+batch_size]

    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        all_logits.append(logits.cpu())

trainer.save_model("/models/camembert/")


# Concatenate logits and compute predictions
test_pred = torch.cat(all_logits, dim=0).numpy()
y_pred_class = np.argmax(test_pred, axis=1)

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Evaluation
print(classification_report(y_test, y_pred_class))

cm = confusion_matrix(y_test, y_pred_class)
sns.heatmap(cm, cmap='Blues', annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()