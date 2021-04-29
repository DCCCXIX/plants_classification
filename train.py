#!/usr/bin/env python
# coding: utf-8

import copy
import logging
import os
import random
import sys
import time
import warnings
from xml.dom import minidom

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
from albumentations import (
    CoarseDropout,
    Compose,
    Cutout,
    HorizontalFlip,
    HueSaturationValue,
    IAAAdditiveGaussianNoise,
    MedianBlur,
    Normalize,
    RandomBrightnessContrast,
    Resize,
    ShiftScaleRotate,
)
from albumentations.pytorch import ToTensorV2
from catalyst.data.sampler import BalanceClassSampler
from sklearn.model_selection import train_test_split
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm


def init_script():
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.INFO)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_train_test_valid_dataloaders(data_path, seed, image_size, batch_size, quality_threshold):
    transform = Compose(
        [
            Resize(width=image_size, height=image_size),
            HorizontalFlip(p=0.4),
            ShiftScaleRotate(p=0.3),
            MedianBlur(blur_limit=7, always_apply=False, p=0.3),
            IAAAdditiveGaussianNoise(scale=(0, 0.15 * 255), p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.4),
            RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            # in this implementation imagenet normalization is used
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.4),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    test_transform = Compose(
        [
            Resize(width=image_size, height=image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )
    """
    Utility function for the model.
    """
    image_paths = []
    labels = []
    image_qualities = []

    for xmldoc in tqdm(os.listdir(data_path)):
        if xmldoc.__contains__(".xml"):
            with open(data_path + xmldoc, "r", encoding="UTF-8") as f:
                doc = f.read()
            doc = minidom.parseString(doc)

            label = doc.getElementsByTagName("ClassId")
            label = int(label[0].firstChild.nodeValue)

            image_path = doc.getElementsByTagName("FileName")
            image_path = image_path[0].firstChild.nodeValue

            image_quality = doc.getElementsByTagName("Vote")
            image_quality = int(image_quality[0].firstChild.nodeValue)

            if image_quality > quality_threshold:
                image_paths.append(data_path + image_path)
                labels.append(label)
                image_qualities.append(image_quality)

    dataset = pd.DataFrame({"image_path": image_paths, "label": labels, "image_quality": image_quality})

    train_dataset, test_dataset = train_test_split(dataset, shuffle=True, test_size=0.1, random_state=seed)
    train_dataset, valid_dataset = train_test_split(train_dataset, shuffle=True, test_size=0.2, random_state=seed)
    train_dataset_labels = train_dataset["label"].values

    train_dataset, valid_dataset = Dataset(train_dataset, transforms=transform), Dataset(
        valid_dataset, transforms=transform
    )
    test_dataset = Dataset(test_dataset, transforms=test_transform)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=BalanceClassSampler(labels=train_dataset_labels, mode="upsampling"),
        # sampler = RandomSampler(train_dataset),
        batch_size=batch_size,
    )
    valid_dataloader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=1)

    return train_dataloader, test_dataloader, valid_dataloader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms=None):
        super().__init__()
        self.image_paths = dataset["image_path"].values
        self.labels = dataset["label"].values
        self.image_qualities = dataset["image_quality"].values
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image, label = self.image_paths[index], self.labels[index]
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        if image.shape[-1] == 4:
            # removing alpha channel if present
            image = image[..., :3]
        if len(image.shape) == 2:
            # converting single channel image to 3 channel for possible greyscales
            image = np.stack((image,) * 3, axis=-1)

        if self.transforms is not None:
            transformed = self.transforms(image=image)

            image = transformed["image"]

        return image, label


class Classifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)

        return x


class Brain(object):
    def __init__(self, gradient_accum_steps=5, lr=0.0005, epochs=100, n_class=500):
        self.device = self.set_cuda_device()
        self.net = Classifier("tf_efficientnet_b3_ns", n_class).to(self.device)
        self.loss_function = nn.MSELoss()
        self.clf_loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.999)
        self.scheduler = CosineAnnealingLR(self.optimizer, epochs * 0.25, lr * 0.0001)
        self.scheduler.last_epoch = epochs
        self.scaler = GradScaler()
        self.epochs = epochs
        self.gradient_accum_steps = gradient_accum_steps

    @staticmethod
    def set_cuda_device():
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logging.info(f"Running on {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logging.info("Running on a CPU")
        return device

    def run_training_loop(self, train_dataloader, valid_dataloader, model_filename):
        best_loss = float("inf")

        for epoch in range(self.epochs):
            if epoch != 0 and epoch > 0.25 * self.epochs:  # cosine anneal the last 25% of epochs
                self.scheduler.step()
            logging.info(f"Epoch {epoch+1}")

            logging.info("Training")
            train_losses, train_accuracies = self.forward_pass(train_dataloader, train=True)

            logging.info("Validating")
            val_losses, val_accuracies = self.forward_pass(valid_dataloader)

            logging.info(
                f"Training accuracy:   {sum(train_accuracies)/len(train_accuracies):.2f} | Training loss: {sum(train_losses)/len(train_losses):.2f}"
            )
            logging.info(
                f"Validation accuracy: {sum(val_accuracies)/len(val_accuracies):.2f} | Validation loss: {sum(val_losses)/len(val_losses):.2f}"
            )

            epoch_val_loss = sum(val_losses) / len(val_losses)

            if best_loss > epoch_val_loss:
                best_loss = epoch_val_loss
                best_model_wts = copy.deepcopy(self.net.state_dict())
                torch.save(self.net.state_dict(), "best.pth")
                logging.info(f"Saving with loss of {epoch_val_loss}, improved over previous {best_loss}")

    def forward_pass(self, dataloader, train=False):
        if train:
            self.net.train()
        else:
            self.net.eval()

        losses = []
        accuracies = []

        for step, batch in enumerate(dataloader):
            inputs = batch[0].to(self.device).float()
            labels = batch[1].to(self.device).long()

            with autocast():
                if train:
                    outputs = self.net(inputs)
                    loss = self.loss_function(outputs, labels)
                    self.scaler.scale(loss).backward()
                else:
                    with torch.no_grad():
                        outputs = self.net(inputs)
                        loss = self.loss_function(outputs, labels)

            matches = [torch.argmax(i) == j for i, j in zip(outputs, labels)]
            acc = matches.count(True) / len(matches)

            losses.append(loss)
            accuracies.append(acc)

            if train and (step + 1) % self.gradient_accum_steps == 0:
                # gradient accumulation to train with bigger effective batch size
                # with less memory use
                # fp16 is used to speed up training and reduce memory consumption
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                logging.info(
                    f"Step {step} of {len(dataloader)},\t"
                    f"Accuracy: {sum(accuracies)/len(accuracies):.2f},\t"
                    f"Loss: {sum(losses)/len(losses):.2f}"
                )

        return losses, accuracies


def main():

    main_ret_status = 0

    SEED = 55555
    init_script()
    seed_everything(seed=SEED)

    train_dataloader, test_dataloader, valid_dataloader = get_train_test_valid_dataloaders(
        data_path=r"./train/", seed=SEED, image_size=300, batch_size=16, quality_threshold=2
    )

    model_filename = "best.pth"
    brain = Brain(gradient_accum_steps=5, lr=0.0005, epochs=100, n_class=2)
    brain.run_training_loop(
        train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, model_filename=model_filename
    )

    brain.net.load_state_dict(torch.load(model_filename))

    start = time.time()
    test_losses, test_accuracies, test_miou = brain.forward_pass(dataloader=test_dataloader, draw=False, train=False)
    total_time = time.time() - start

    logging.info(f"Average inference time is: {total_time/len(test_dataloader):.3f}")
    logging.info(
        f"Test accuracy: {sum(test_accuracies)/len(test_accuracies):.2f}"
        f" | Test loss: {sum(test_losses)/len(test_losses):.2f} | Test mIoU: {sum(test_miou)/len(test_miou):.2f}"
    )

    return main_ret_status


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
