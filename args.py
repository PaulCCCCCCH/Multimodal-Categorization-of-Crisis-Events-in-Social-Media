import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--batch_size', default=1, type=int)

    # model
    parser.add_argument('--model', default='resnet', type=str)

    # data
    parser.add_argument('--load_size', default=228, type=int)
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--max_dataset_size', default=2147483648, type=int)

    return parser.parse_args()
