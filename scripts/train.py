import os
import argparse

import context
from visualdet3d.models.utils.registry import AUGMENTATION_DICT, DATASET_DICT, DETECTOR_DICT, PIPELINE_DICT
from visualdet3d.optim.optimizers import get_optimizer
from visualdet3d.optim.schedulers import get_scheduler
from configs import load_config


def main():
    # Load config
    cfg = load_config(args.config)

    # Create dataset
    train_loader = DATASET_DICT[cfg.data.train_dataset](cfg, split='training')
    val_loader = DATASET_DICT[cfg.data.val_dataset](cfg, split='validation')

    # Build model
    model = DETECTOR_DICT[cfg.detector.name](cfg.detector)

    # Build optimizer and scheduler
    lr_scheduler = get_scheduler(cfg)
    optimizer = get_optimizer(lr_scheduler, cfg)

    if 'training_func' in cfg.trainer:
        training_dection = PIPELINE_DICT[cfg.trainer.training_func]
        print(f'Found training function {cfg.trainer.training_func}')
    else:
        raise KeyError

    # Get evaluation pipeline
    if 'evaluate_func' in cfg.trainer:
        evaluate_detection = PIPELINE_DICT[cfg.trainer.evaluate_func]
        print(f'Found evaluate function {cfg.trainer.evaluate_func}')
    else:
        evaluate_detection = None
        print('Evaluate function not found')

    # Train
    global_step = 0
    for epoch in range(cfg.trainer.max_epochs):
        for idx, data in enumerate(train_loader):
            loss = training_dection(data, model, optimizer, cfg=cfg)
            global_step += 1
            if global_step % cfg.trainer.disp_iter == 0:
                print(f'[Epoch {epoch+1:03d} iter {idx+1:04d}] Loss: {loss.numpy():.4f}')
        
            if global_step % cfg.trainer.save_iter == 0:
                if not os.path.isdir(args.save_dir):
                    os.mkdir(args.save_dir)
                save_path = os.path.join(args.save_dir, 'model.ckpt')
                model.save_weights(save_path)
                print(f'Saved model as {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--save_dir', type=str, default='./saved_models/',
                        help='Path to save model checkpoint')
    args = parser.parse_args()
    main()