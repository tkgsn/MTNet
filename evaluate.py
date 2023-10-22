# evaluation using evaluate function in ../../priv_traj_gen/run.py
import sys
import pathlib
from logging import getLogger
import json
import numpy as np
sys.path.append('../../priv_traj_gen')
from run import evaluate
from my_utils import load
from dataset import TrajectoryDataset


class MockGenerator():
    '''
    This generator returns the trajectory in the given path
    '''
    def __init__(self, path):
        # load
        self.trajs = load(path)
        self.cursor = -1

    def eval(self):
        pass

    def train(self):
        pass

    def make_sample(self, references, mini_batch_size):
        '''
        return the mini_batch_size trajectories in the given path
        '''
        self.cursor += 1
        return self.trajs[self.cursor*mini_batch_size:(self.cursor+1)*mini_batch_size]

class Namespace():
    pass

def set_args():

    args = Namespace()
    args.evaluate_global = False
    args.evaluate_passing = True
    args.evaluate_source = True
    args.evaluate_target = True
    args.evaluate_route = True
    args.evaluate_destination = True
    args.evaluate_distance = True
    args.evaluate_first_next_location = False
    args.evaluate_second_next_location = False
    args.evaluate_second_order_next_location = False
    args.eval_initial = True
    args.eval_interval = 1
    args.n_test_locations = 30
    args.dataset = "geolife"
    args.n_split = 5
    # this is not used
    args.batch_size = 100

    return args

def run(generated_data_path, original_training_data_path, save_path):

    generated_data_path = pathlib.Path(generated_data_path)
    original_training_data_path = pathlib.Path(original_training_data_path)
    save_path = pathlib.Path(save_path)

    logger = getLogger(__name__)
    trajectories = load(original_training_data_path / "training_data.csv")
    time_trajectories = load(original_training_data_path / "training_data_time.csv")
    logger.info(f"load training data from {original_training_data_path / 'training_data.csv'}")
    logger.info(f"load time data from {original_training_data_path / 'training_data_time.csv'}")

    args = set_args()

    # load setting file
    with open(original_training_data_path / "params.json", "r") as f:
        param = json.load(f)
    n_bins = param["n_bins"]
    n_locations = (n_bins+2)**2

    dataset = TrajectoryDataset(trajectories, time_trajectories, n_locations, args.n_split)
    dataset.compute_auxiliary_information(save_path, logger)

    # find the generated data in the given generated_data_path
    files = [file for file in generated_data_path.iterdir() if file.name.startswith("generated_")]

    # evaluation
    resultss = []
    for i, file in enumerate(files):
        print(i)
        logger.info(f"evaluate {file}")
        generator = MockGenerator(file)
        epoch = i+1
        results = evaluate(generator, dataset, args, epoch)
        resultss.append(results)
    with open(save_path / "results.json", "w") as f:
        json.dump(resultss, f)


if __name__ == "__main__":
    generated_data_path = sys.argv[1]
    original_training_data_path = sys.argv[2]
    save_path = sys.argv[3]
    run(generated_data_path, original_training_data_path, save_path)