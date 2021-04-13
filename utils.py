import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 100
ROOT = 'F:/ear_classifier_demo_1/data'


if __name__ == '__main__':
    with open('F:/ear_classifier_demo_1/data/difficult_train.txt', 'r') as f:
        diffc_label = f.read().split('\n')
    diffc_label.pop(-1)
    diffc_label = [int(x) for x in diffc_label]
    print(diffc_label)