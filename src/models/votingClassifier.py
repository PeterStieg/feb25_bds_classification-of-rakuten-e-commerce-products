from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import pickle
import numpy as np
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score  # Import f1_score
import argparse

# import files from gogoel drive
# This part might need to be adjusted if running outside Colab
# from google.colab import drive
# drive.mount('/content/drive')


# Preprocessing Data
class CustomImageTextDataset(Dataset):
    def __init__(self, df, image_dir, tokenizer, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __getitem__(self, idx):
        # Load image
        image_name = self.df.iloc[idx]["image_name"]
        image_path = os.path.join(self.image_dir, f"{image_name}")
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Load and tokenize text
        text = self.df.iloc[idx]["text"]

        # Ensure text is a string and handle missing values
        if pd.isna(text):
            text = ""
        else:
            text = str(text)

        # Tokenize the text data with padding to max_length
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",  # Return pytorch tensors
            padding="max_length",  # Pad to a fixed max length
            truncation=True,  # Truncate if text is longer than max_length
            max_length=256,  # Set a fixed max length for all sequences
        )

        # Squeeze the batch dimension
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}

        # Load label
        label = self.df.iloc[idx]["prdtypecode_encoded"]

        return image, text_inputs, label

    def __len__(self):
        return len(self.df)

def predict_voting(image, text, raw_texts, model_vgg16, model_bert, svm_model, vectorizer, le, device):
    # ----- Image prediction -----
    with torch.no_grad():
        image = image.to(device)
        image_output = model_vgg16(image)  # Get output for the image (logits)
        image_prob = F.softmax(image_output, dim=1)  # Convert logits to probabilities

    # ----- Text prediction (BERT) -----
    with torch.no_grad():
        texts = {key: val.to(device) for key, val in text.items()}
        text_output = model_bert(**texts)  # Get output for the text (logits)
        text_prob = F.softmax(text_output.logits, dim=1)  # Convert logits to probabilities
        
    # ----- Text prediction (SVM) -----
    with torch.no_grad():
        # raw_texts is a list of strings
        text_features = vectorizer.transform(raw_texts)  # Use saved vectorizer
        svm_text_prob_np = svm_model.predict_proba(text_features)  # shape: (batch_size, num_classes)

        # Convert to torch tensor for voting
        svm_text_prob = torch.tensor(svm_text_prob_np, dtype=torch.float32).to(device)

    # ----- Combine the probabilities -----
    avg_prob = (image_prob + text_prob + svm_text_prob) / 3.0

    # Final prediction
    final_prediction_encoded = torch.argmax(avg_prob, dim=1).cpu().numpy()
    final_prediction_decoded = le.inverse_transform(final_prediction_encoded)

    # Individual predictions
    image_prediction_decoded = le.inverse_transform(torch.argmax(image_prob, dim=1).cpu().numpy())
    text_prediction_decoded = le.inverse_transform(torch.argmax(text_prob, dim=1).cpu().numpy())
    svc_text_prediction_decoded = le.inverse_transform(torch.argmax(svm_text_prob, dim=1).cpu().numpy())

    return final_prediction_decoded, image_prediction_decoded, text_prediction_decoded, svc_text_prediction_decoded, avg_prob



def main():
    parser = argparse.ArgumentParser(
        description="Run predictions on a sample or calculate F1 score."
    )
    parser.add_argument(
        "--run_type",
        type=str,
        default="sample",
        choices=["sample", "f1"],
        help='Specify run type: "sample" for random sample, "f1" for F1 score calculation.',
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=30,
        help='Size of the random sample (only applicable when run_type is "sample").',
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Batch size for DataLoader."
    )
    args = parser.parse_args()

    # Constants
    # Adjust paths as necessary if not running in the same Colab environment
    image_dir = "images/image_train"
    df_path = "language_analysis/df_localization.csv"
    camembert_model_path = "models/bert"
    vgg16_model_path = "models/vgg16_transfer_model.pth"

    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset
    df = pd.read_csv(df_path)
    # create a text column with translations where necessary
    df["text"] = np.where(
        df["deepL_translation"].notna(),
        df["deepL_translation"],
        np.where(df["lang"] == "fr", df["merged_text"], np.nan),
    )

    df = df[["prdtypecode", "productid", "imageid", "text"]]
    df["image_name"] = (
        "image_"
        + df["imageid"].astype(str)
        + "_product_"
        + df["productid"].astype(str)
        + ".jpg"
    )
    df.drop(["productid", "imageid"], axis=1, inplace=True)

    # The camemBERT model used LabelEncoder()
    le = LabelEncoder()
    # Fit and transform the prdtypecode column
    df["prdtypecode_encoded"] = le.fit_transform(df["prdtypecode"])

    df_train, df_test = train_test_split(
        df, random_state=42, stratify=df["prdtypecode_encoded"], test_size=0.2
    )
    df_test.head()

    # Define a function that checks if the image file exists
    def image_exists(row, image_dir):
        image_path = os.path.join(image_dir, f"{row['image_name']}")
        return os.path.exists(image_path)

    # Apply the function and filter the DataFrame
    # Note: This filtering might take time depending on the number of images and drive speed
    # print("Filtering df_test to include only existing images...")
    # df_test = df_test[df_test.apply(image_exists, axis=1, args=(image_dir,))]
    print(f"Filtered df_test size: {df.shape[0]}")

    df_test.dropna(inplace=True)

    # Load Camembert model
    from transformers import CamembertForSequenceClassification, CamembertTokenizer

    model_bert = CamembertForSequenceClassification.from_pretrained(
        camembert_model_path
    )
    tokenizer = CamembertTokenizer.from_pretrained(camembert_model_path)


    # Move BERT model to the specified device
    model_bert.to(device)

    # Load Classical Text Model

    import joblib
    vectorizer = joblib.load('models/svc_vectorizer.pkl')  # CPU-based model
    svm_model = joblib.load('models/svm_classifier.pkl')  # CPU-based model

    # Load VGG16 model
    model_vgg16 = models.vgg16(pretrained=True)
    model_vgg16.classifier[6] = torch.nn.Linear(in_features=4096, out_features=27)
    model_vgg16.load_state_dict(
        torch.load(vgg16_model_path, map_location=device)
    )  # Add map_location
    model_vgg16.eval()  # Set to evaluation mode

    # Move VGG16 model to the specified device
    model_vgg16.to(device)

    # Image Transformation (for VGG16)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # VGG16 specific normalization
        ]
    )

    # Set Up the Dataset
    dataset = CustomImageTextDataset(df_test, image_dir, tokenizer, transform=transform)

    if args.run_type == "sample":
        if args.sample_size > len(dataset):
            print(
                f"Warning: Sample size ({args.sample_size}) is larger than the dataset size ({len(dataset)}). Using the entire dataset size as sample size."
            )
            sample_size = len(dataset)
        else:
            sample_size = args.sample_size

        # Take a random sample of indices from the test set
        sample_indices = np.random.choice(len(dataset), size=sample_size, replace=False)
        sample_dataset = torch.utils.data.Subset(
            dataset, sample_indices
        )  # Use Subset for sampling

        # Create a DataLoader for the sample
        sample_data_loader = DataLoader(
            sample_dataset, batch_size=args.batch_size, shuffle=False
        )

        print(
            f"Running predictions on a random sample of {sample_size} items in batches of {args.batch_size}:"
        )

        # Iterate through the sample data loader
        for batch_idx, (images, texts, labels_encoded) in enumerate(sample_data_loader):
            # Extract raw texts for this batch from tokenizer input
            raw_texts = [tokenizer.decode(texts['input_ids'][i], skip_special_tokens=True) for i in range(len(images))]
            
            # Decode the actual labels
            actual_labels_decoded = le.inverse_transform(labels_encoded.cpu().numpy())
            # Get predictions from the voting classifier and individual models
            final_pred_decoded, image_pred_decoded, text_pred_decoded,svm_pred_decoded, avg_prob = (
                predict_voting(images, texts, raw_texts, model_vgg16, model_bert, svm_model, vectorizer, le, device)
            )

            # Print the results for each item in the sample
            print("\n--- Sample Prediction Results ---")
            for i in range(len(actual_labels_decoded)):
                print(f"Item {i+1}:")
                print(f"  Actual Label: {actual_labels_decoded[i]}")
                print(f"  VGG16 Prediction: {image_pred_decoded[i]}")
                print(f"  BERT Prediction: {text_pred_decoded[i]}")
                print(f"  SVC Prediction: {svm_pred_decoded[i]}")
                print(f"  Voting Prediction: {final_pred_decoded[i]}")
                # Optional: Print probabilities for the voting prediction
                # print(f"  Voting Probabilities: {avg_prob[i].tolist()}")
            print("---------------------------------")

    elif args.run_type == "f1":
        from tqdm import tqdm

        # Create a DataLoader for the entire test set
        full_test_data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False
        )

        all_actual_labels = []
        all_voting_predictions = []

        print(
            f"Running predictions on the entire df_test dataset ({len(dataset)} items) to calculate F1 score with batch size {args.batch_size}..."
        )

        # Iterate through the full test data loader
        for images, texts, labels_encoded in tqdm(
            full_test_data_loader, desc="Predicting", unit="batch"
        ):
            # Get predictions from the voting classifier
            final_pred_decoded, _, _, _ = predict_voting(
                images, texts, raw_texts, model_vgg16, model_bert, svm_model, vectorizer, le, device
            )

            # Store actual labels and voting predictions
            all_actual_labels.extend(
                labels_encoded.cpu().numpy()
            )  # Store encoded labels for f1_score
            all_voting_predictions.extend(
                le.transform(final_pred_decoded)
            )  # Store encoded predictions

        print("Finished making predictions.")

        # Calculate the F1 score
        f1 = f1_score(all_actual_labels, all_voting_predictions, average="weighted")

        print(f"\nWeighted F1 Score on df_test: {f1}")


if __name__ == "__main__":
    # If you are running this script in a non-Colab environment,
    # you might need to adjust how you access files from Google Drive.
    # One common approach is to download them first.
    # If running in Colab, the drive.mount() at the top will handle this.

    main()
