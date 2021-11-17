import argparse


def get_args():
    parser = argparse.ArgumentParser()


    # -------------- Important configs --------------- #
    parser.add_argument('--mode', choices=['both', 'image_only', 'text_only'])
    parser.add_argument('--task', choices=['task1', 'task2'])
    parser.add_argument('--learning_rate', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--with_sse', action='store_true')
    parser.add_argument('--save_dir', default='./output', type=str)
    parser.add_argument('--model_name', default='', type=str)


    # only used when with_sse set
    parser.add_argument('--pv', default=1000, type=int)
    parser.add_argument('--pt', default=1000, type=int)
    parser.add_argument('--pv0', default=0.3, type=float)
    parser.add_argument('--pt0', default=0.7, type=float)

    # Loading model 
    parser.add_argument('--model_to_load', default='')
    parser.add_argument('--image_model_to_load', default='')
    parser.add_argument('--text_model_to_load', default='')


    parser.add_argument('--max_iter', default=300, type=int)

    # -------------- Default ones --------------- #
    # Running configs

    # Run flag
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--use_tensorboard', action='store_true')

    # System configs
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', default=0, type=int)


    # data processing
    parser.add_argument('--load_size', default=228, type=int)
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--max_dataset_size', default=2147483648, type=int)


    
    return parser.parse_args()
