from utils.dataloader import load_data, data_loader,tokenize,load_pretrained_vectors
import torch.optim as optim
from model.cnn_nlp import CNN_NLP
#from model.cnn_model import CNN_NLP2
from sklearn.metrics import classification_report
#from ignite.metrics import ClassificationReport
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import random
import time
import argparse

def initilize_model(pretrained_embedding=None,
                    freeze_embedding=False,
                    vocab_size=None,
                    embed_dim=300,
                    filter_sizes=[3, 4, 5], #[3,8],
                    num_filters=[100, 100, 100], #[128, 128]
                    num_classes=2,
                    dropout=0.5,
                    learning_rate=0.01):
    """Instantiate a CNN model and an optimizer."""

    assert (len(filter_sizes) == len(num_filters)), "filter_sizes and \
    num_filters need to be of the same length."

    # Instantiate CNN model
    cnn_model = CNN_NLP(pretrained_embedding=pretrained_embedding,
                        freeze_embedding=freeze_embedding,
                        vocab_size=vocab_size,
                        embed_dim=embed_dim,
                        filter_sizes=filter_sizes,
                        num_filters=num_filters,
                        num_classes=2,
                        dropout=0.5)

    # Send model to `device` (GPU/CPU)
    cnn_model.to(device)

    # Instantiate Adadelta optimizer
    optimizer = optim.Adadelta(cnn_model.parameters(),
                               lr=learning_rate,
                               rho=0.95)

    return cnn_model, optimizer


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, optimizer, train_dataloader, val_dataloader=None, epochs=10):
    """Train the CNN model."""

    # Tracking best validation accuracy
    best_accuracy = 0

    # Start training loop
    print("Start training...\n")
    #print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | { 'Val Acc': ^ 9} | {'Elapsed': ^ 9}")
    print(f"{'Epoch'} | {'Train Loss'} | {'Val Loss'} | {'Val Acc'} | {'Elapsed'}")
    print("-" * 60)

    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================

        # Tracking time and loss
        t0_epoch = time.time()
        total_loss = 0

        # Put the model into the training mode
        model.train()

        for step, batch in enumerate(train_dataloader):
            # Load batch to GPU
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Update parameters
            optimizer.step()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        # =======================================
        #               Evaluation
        # =======================================
        if val_dataloader is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Track the best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | { val_loss: ^ 10.6f} | {val_accuracy: ^ 9.2f} | {time_elapsed: ^ 9.2f}")

    print("\n")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")

def get_classification_report(y_test, preds):
    cr = classification_report(y_test, preds , output_dict=True)
    return pd.DataFrame(cr).transpose()

def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # metric = ClassificationReport(output_dict=True)
    # metric.attach(default_evaluator, "cr")

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # print(preds)
        # print(type(preds))
        # print(b_labels)
        # print(type(b_labels))

        pred_arr = preds.cpu().detach().numpy()
        b_labels_arr = b_labels.cpu().detach().numpy()
        print(get_classification_report(b_labels_arr,pred_arr))

        # state = default_evaluator.run([[preds, b_labels]])
        # print(state.metrics["cr"].keys())
        # print(state.metrics["cr"]["0"])
        # print(state.metrics["cr"]["1"])
        # print(state.metrics["cr"]["macro avg"])


        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


def save_model(model, model_dir):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(model_dir, "model.pt")
        torch.save(model, model_path)

def save_vocab(vocab, model_dir):
    """Save vocab file to `model_dir` directory"""
    vocab_path = os.path.join(model_dir, "vocab.pt")
    torch.save(vocab, vocab_path)

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='path to csv files')
    parser.add_argument('--embedding_dir', type=str, required=True, help='path to embedding directory')
    parser.add_argument('--model_dir', type=str, required=True, help='path to test csv files')
    args = parser.parse_args()

    #data_dir = '../data/merged_split/1960/*.csv'
    data_dir = args.data_dir
    embedding_dir = args.embedding_dir
    model_dir = args.model_dir

    df = load_data(data_dir)
    encoded_texts, vocab = tokenize(df['text'])

    # Train Test Split
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        encoded_texts, df['labels'], test_size=0.1, random_state=42)

    # print(train_inputs)
    # print(type(val_inputs))
    # print(type(train_labels))
    # print(type(val_labels))
    #
    # s = CNN2_model(args.nrow_input, args.ncol_input, args.num_channels, args.epochs, args.batch_size,
    #                args.channel_first)
    # print(" X_train.shpe", X_train.shape)
    # print(" y_train.shpe", y_train.shape)
    # print(" X_test.shpe", X_test.shape)
    # print(" y_test.shpe", y_test.shape)
    #
    # s.apply_model(X_train, y_train, X_test, y_test, args.output_dir)
    # s.save_results(y_test, args.output_dir)  # , cv_results)

    ###### cnn pytorch
    # Load data to PyTorch DataLoader
    train_dataloader, val_dataloader = data_loader(train_inputs, val_inputs, train_labels, val_labels, batch_size=50)

    #embedding_dir = '../weights/cbow_histaware_63/'
    embeddings = load_pretrained_vectors(embedding_dir,vocab)
    #

    loss_fn = nn.CrossEntropyLoss()

    set_seed(42)
    cnn_non_static, optimizer = initilize_model(pretrained_embedding=embeddings, #vocab_size=len(vocab), embed_dim=300,
                                          freeze_embedding=False,
                                          learning_rate=0.25,
                                          dropout=0.5)

    train(cnn_non_static, optimizer, train_dataloader, val_dataloader, epochs=2)

    save_model(cnn_non_static, model_dir)
    save_vocab(vocab, model_dir)
    ############ cnn pytorch


