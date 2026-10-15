"""
Handwritten digit recognition (CodeAlpha internship, Task 3)

This script trains a small Convolutional Neural Network (CNN) on the
MNIST dataset. MNIST has 70,000 pictures of handwritten digits (0 to 9),
each picture is 28 x 28 pixels in grey.

The training pictures are split into a training part and a validation
part. The validation part helps me watch the learning while it happens,
and the 10,000 test pictures are used only once, at the very end.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATA_FOLDER = Path("data")
RESULTS_FOLDER = Path("results")
RANDOM_SEED = 42
PICTURES_PER_BATCH = 128
NUMBER_OF_EPOCHS = 5
LEARNING_RATE = 0.001
NUMBER_OF_VALIDATION_PICTURES = 6000

# Average and spread of the MNIST pixel values. Used to scale the pictures.
MNIST_PIXEL_AVERAGE = 0.1307
MNIST_PIXEL_SPREAD = 0.3081


class HandwrittenDigitCnn(nn.Module):
    def __init__(self):
        super().__init__()

        # Two rounds of "look for small patterns, then shrink the picture".
        self.picture_pattern_finder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 x 28 becomes 14 x 14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 x 14 becomes 7 x 7
        )

        # Turn the patterns into one score for each digit from 0 to 9.
        self.digit_chooser = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),  # randomly switch off some values so it does not memorise
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, batch_of_pictures):
        patterns = self.picture_pattern_finder(batch_of_pictures)
        return self.digit_chooser(patterns)


def load_mnist_data():
    picture_preparation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_PIXEL_AVERAGE,), (MNIST_PIXEL_SPREAD,)),
        ]
    )

    full_training_set = datasets.MNIST(
        DATA_FOLDER, train=True, download=True, transform=picture_preparation
    )
    test_set = datasets.MNIST(
        DATA_FOLDER, train=False, download=True, transform=picture_preparation
    )

    number_of_training_pictures = len(full_training_set) - NUMBER_OF_VALIDATION_PICTURES
    split_generator = torch.Generator().manual_seed(RANDOM_SEED)
    training_set, validation_set = random_split(
        full_training_set,
        [number_of_training_pictures, NUMBER_OF_VALIDATION_PICTURES],
        generator=split_generator,
    )

    training_loader = DataLoader(training_set, batch_size=PICTURES_PER_BATCH, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=PICTURES_PER_BATCH)
    test_loader = DataLoader(test_set, batch_size=PICTURES_PER_BATCH)
    return training_loader, validation_loader, test_loader


def train_for_one_epoch(model, training_loader, loss_function, optimizer, device):
    model.train()
    total_loss = 0.0

    for pictures, real_digits in training_loader:
        pictures = pictures.to(device)
        real_digits = real_digits.to(device)

        optimizer.zero_grad()
        digit_scores = model(pictures)
        loss = loss_function(digit_scores, real_digits)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(pictures)

    return total_loss / len(training_loader.dataset)


def collect_predictions(model, data_loader, device):
    model.eval()
    real_digits_list = []
    predicted_digits_list = []

    with torch.no_grad():
        for pictures, real_digits in data_loader:
            digit_scores = model(pictures.to(device))
            predicted_digits_list.extend(digit_scores.argmax(dim=1).cpu().tolist())
            real_digits_list.extend(real_digits.tolist())

    return np.array(real_digits_list), np.array(predicted_digits_list)


def save_learning_curves_picture(loss_per_epoch, validation_accuracy_per_epoch):
    epoch_numbers = range(1, len(loss_per_epoch) + 1)

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(10, 4))
    loss_axis.plot(epoch_numbers, loss_per_epoch, marker="o")
    loss_axis.set_title("Training loss")
    loss_axis.set_xlabel("Epoch")

    accuracy_axis.plot(epoch_numbers, validation_accuracy_per_epoch, marker="o")
    accuracy_axis.set_title("Validation accuracy")
    accuracy_axis.set_xlabel("Epoch")

    figure.tight_layout()
    figure.savefig(RESULTS_FOLDER / "learning_curves.png", dpi=150)
    plt.close(figure)


def save_confusion_matrix_picture(real_digits, predicted_digits):
    counts = confusion_matrix(real_digits, predicted_digits)

    plt.figure(figsize=(7, 6))
    plt.imshow(counts, cmap="Blues")
    plt.colorbar()
    plt.title("Confusion matrix on the test pictures")
    plt.xlabel("Predicted digit")
    plt.ylabel("Real digit")
    plt.xticks(range(10))
    plt.yticks(range(10))

    biggest_count = counts.max()
    for real_digit in range(10):
        for predicted_digit in range(10):
            number = counts[real_digit, predicted_digit]
            text_colour = "white" if number > biggest_count / 2 else "black"
            plt.text(predicted_digit, real_digit, str(number),
                     ha="center", va="center", color=text_colour, fontsize=7)

    plt.tight_layout()
    plt.savefig(RESULTS_FOLDER / "confusion_matrix.png", dpi=150)
    plt.close()


def save_wrong_predictions_picture(test_set, real_digits, predicted_digits):
    wrong_positions = np.where(real_digits != predicted_digits)[0][:12]
    if len(wrong_positions) == 0:
        return

    figure, axes = plt.subplots(3, 4, figsize=(8, 6))
    for axis in axes.flat:
        axis.axis("off")

    for axis, position in zip(axes.flat, wrong_positions):
        picture, _ = test_set[position]
        axis.imshow(picture.squeeze(), cmap="gray")
        axis.set_title(f"real {real_digits[position]}, guessed {predicted_digits[position]}",
                       fontsize=9)

    figure.suptitle("Some pictures the model got wrong")
    figure.tight_layout()
    figure.savefig(RESULTS_FOLDER / "wrong_predictions.png", dpi=150)
    plt.close(figure)


def main():
    RESULTS_FOLDER.mkdir(exist_ok=True)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    training_loader, validation_loader, test_loader = load_mnist_data()
    print(f"Training pictures: {len(training_loader.dataset)}, "
          f"validation pictures: {len(validation_loader.dataset)}, "
          f"test pictures: {len(test_loader.dataset)}")

    model = HandwrittenDigitCnn().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_per_epoch = []
    validation_accuracy_per_epoch = []

    for epoch_number in range(1, NUMBER_OF_EPOCHS + 1):
        average_loss = train_for_one_epoch(
            model, training_loader, loss_function, optimizer, device
        )
        real_digits, predicted_digits = collect_predictions(model, validation_loader, device)
        validation_accuracy = accuracy_score(real_digits, predicted_digits)

        loss_per_epoch.append(average_loss)
        validation_accuracy_per_epoch.append(validation_accuracy)
        print(f"Epoch {epoch_number}/{NUMBER_OF_EPOCHS}: "
              f"training loss {average_loss:.4f}, "
              f"validation accuracy {validation_accuracy:.4f}")

    # The test pictures are used only now, once, after all the training.
    real_digits, predicted_digits = collect_predictions(model, test_loader, device)
    test_accuracy = accuracy_score(real_digits, predicted_digits)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Wrong test pictures: {(real_digits != predicted_digits).sum()} out of {len(real_digits)}")
    print("\nResult for each digit:")
    print(classification_report(real_digits, predicted_digits, digits=3))

    save_learning_curves_picture(loss_per_epoch, validation_accuracy_per_epoch)
    save_confusion_matrix_picture(real_digits, predicted_digits)
    save_wrong_predictions_picture(test_loader.dataset, real_digits, predicted_digits)
    torch.save(model.state_dict(), RESULTS_FOLDER / "handwritten_digit_cnn.pt")
    print("\nSaved the pictures and the trained model in the results folder.")


if __name__ == "__main__":
    main()