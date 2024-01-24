import os
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import hydra
from omegaconf import DictConfig
import random
import colorsys
import os
from ..data import SemanticSegmentationDataset
import json
import yaml
import csv


def compare_models(cfg: DictConfig) -> None:
    '''
    Generates csv file with metrics computed for all the models on validation set. Furthermore it produces plots depicting
    change of training loss and evaluation metrics over the course of finetuning.
    '''
    output_dir = '/' + os.path.join(*((hydra.core.hydra_config.HydraConfig.get().runtime.output_dir).split('/')[:-2]))
    day_dirs = os.listdir(output_dir)
    csv_file = open('models_comparison.csv', 'w')
    csv_writer = csv.writer(csv_file)
    no_header = True
    for day_dir in day_dirs:
        experiment_dirs = os.listdir(os.path.join(output_dir, day_dir))
        for experiment_dir in experiment_dirs:
            if 'finetuned_model' in os.listdir(os.path.join(output_dir, day_dir, experiment_dir)):
                checkpoints = os.listdir(os.path.join(output_dir, day_dir, experiment_dir, 'finetuned_model'))
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))
                latest_checkpoint = checkpoints[-1]

                with open(os.path.join(output_dir, day_dir, experiment_dir, '.hydra', 'config.yaml'), 'r') as file:
                    hydra_config = yaml.safe_load(file)

                file = open(os.path.join(output_dir, day_dir, experiment_dir, 'finetuned_model', latest_checkpoint, 'trainer_state.json')) 
                trainer_state = json.load(file)

                best_model_checkpoint = int(trainer_state['best_model_checkpoint'].split('-')[1])

                metrics = dict()
                for evaluation in trainer_state['log_history']:
                    if 'eval_mean_iou' in evaluation:
                        if evaluation['step'] == best_model_checkpoint:
                            cleared_evaluation = evaluation.copy()
                            for key in ['epoch', 'eval_loss', 'eval_runtime', 'step', 'eval_steps_per_second']:
                                cleared_evaluation.pop(key)

                            if no_header:
                                csv_writer.writerow(['model', 'pretrained model'] + list(cleared_evaluation.keys()))
                                no_header = False
                            csv_writer.writerow([hydra_config['model']['name'], hydra_config['model']['pretrained_model']] + [round(val, 2) for val in list(cleared_evaluation.values())])

                        for metric in evaluation:
                            if metric not in ('epoch', 'eval_loss', 'eval_runtime', 'step', 'eval_samples_per_second', 'eval_steps_per_second'):
                                if metric not in metrics:
                                    metrics[metric] = []
                                metrics[metric].append(evaluation[metric])

                for metric in metrics:
                    plt.plot(metrics[metric], label=metric[5:])
                plt.legend()
                plt.xlabel('epoch')
                plt.ylabel('metric value')
                plt.title('Evaluation on validation set: {} {}'.format(hydra_config['model']['name'], hydra_config['model']['pretrained_model']))
                plt.savefig('eval_{}_{}.png'.format(hydra_config['model']['name'], hydra_config['model']['pretrained_model']).replace('/', '_'))
                plt.clf()

                losses = []
                for v in trainer_state['log_history']:
                    if 'loss' in v:
                        losses.append(v['loss'])
                plt.plot(losses)
                plt.xlabel('steps')
                plt.ylabel('loss')
                plt.title('Training loss: {} {}'.format(hydra_config['model']['name'], hydra_config['model']['pretrained_model']))
                plt.savefig('training_loss_{}_{}.png'.format(hydra_config['model']['name'], hydra_config['model']['pretrained_model']).replace('/', '_'))
                plt.clf()

    csv_file.close()


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    compare_models(cfg)

if __name__ == "__main__":
    main()
