import os
import glob
import argparse
from pathlib import Path

import numpy as np


def create_dataset(out_dir, split, names):
    """
    """
    save_path = os.path.join(out_dir, f'{split}.txt')
    with open(save_path, 'wt') as f:
        for name in names:
            f.write(name + '\n')
    return save_path


def main():
    calib_dir = os.path.join(args.data_dir, 'calib')

    calib_paths = sorted(glob.glob(os.path.join(calib_dir, '*.txt')))

    np.random.seed(12)
    indices = np.arange(len(calib_paths))
    num_train = int(len(calib_paths) * (1 - args.split))

    calib_paths = np.array(calib_paths)[indices].tolist()
    train_calib_paths = calib_paths[:num_train]
    val_calib_paths = calib_paths[num_train:]

    train_names = [Path(path).stem for path in train_calib_paths]
    val_names = [Path(path).stem for path in val_calib_paths]
    print(f'Train/Val: {len(train_names)}/{len(val_names)}')

    out_dir = Path(args.data_dir).parents[0]
    print('Creating training text file')
    save_path = create_dataset(out_dir, 'training', train_names)
    print(f'Saved as {save_path}')

    print('Creating validation text file')
    save_path = create_dataset(out_dir, 'validation', val_names)
    print(f'Saved as {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='Data root directory')
    parser.add_argument('--split', type=float, default=0.2,
                        help='Validation set percentage')
    args = parser.parse_args()
    main()