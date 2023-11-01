import unittest
import json
from make_training_data import make_edges, convert, compensate_edge, run
from convert_to_original_format import convert_to_original_format
from evaluate import run as evaluation
import pathlib
import os
import sys

sys.path.append("../../priv_traj_gen")
from my_utils import plot_density, load
from evaluation import count_route_locations

class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.test_data_path = "/data/results/geolife/0/narrow_0_0_bin30_seed0/MTNet"
        self.original_data_path = "/data/geolife/0/narrow_0_0_bin30_seed0/"
        self.stay_point_data_path = "/data/geolife/0/narrow_200_10_bin30_seed0/"
        self.save_path = "/data/results/geolife/0/narrow_0_0_bin30_seed0/DP_MTNet"

    def test_run(self):
        evaluation(self.test_data_path, self.original_data_path, self.stay_point_data_path, self.save_path)

    def test_plot_density(self):
        path = os.path.join(self.save_path, "generated_80.csv")
        trajs = load(path)
        counter = count_route_locations(trajs, 337)
        plot_density(counter, 32*32, "./data/route_density.png", 337)



class TestConvertToOriginalFormat(unittest.TestCase):
    def setUp(self):
        self.test_data_path = "./data/test/trajs_demo.csv"

    def test_convert(self):
        trajs = convert_to_original_format(self.test_data_path)
        print(trajs[:10])

class TestMakeTrainingData(unittest.TestCase):
    
    def setUp(self):
        self.test_data_path = "/data/geolife/100/narrow_0_0_bin30_seed0/training_data.csv"
        # self.test_data_path = "/data/geolife/1000/narrow_0_0_bin30_seed0/training_data.csv"
        self.save_path = pathlib.Path("./data/test")
        self.save_path.mkdir(parents=True, exist_ok=True)


    def test_run(self):
        save_path = self.save_path
        run(self.test_data_path, save_path)

        test_data_path = save_path / "trajs_demo.csv"
        trajs = []
        with open(test_data_path, "r") as f:
            for line in f:
                trajs.append([int(vocab) for vocab in line.split()])

        # check if the all two consecutive edges are neighbors
        # load id_to_edge.json
        with open(save_path / "id_to_edge.json", "r") as f:
            id_to_edge = json.load(f)
        id_to_edge = {int(k):v for k,v in id_to_edge.items()}
        for traj in trajs:
            for i in range(len(traj)-2):
                edge1 = id_to_edge[traj[i]]
                edge2 = id_to_edge[traj[i+1]]

                if i == 0:
                    # the first place is always the start location
                    self.assertEqual(len(edge1), 1)
                    self.assertEqual(edge1[0], edge2[0])
                else:
                    # print(traj[i], traj[i+1])
                    # print(edge1, edge2)
                    self.assertEqual(len(edge1), 2)
                    self.assertEqual(edge1[1], edge2[0])

        # test the edge_adj.txt file
        with open(save_path / "edge_adj.txt", "r") as f:
            for i, line in enumerate(f):
                to_location = id_to_edge[i+1][-1]

                adj_edges = [int(adj_edge) for adj_edge in line.split(",")[1:]]
                adj_edges = [adj_edge for adj_edge in adj_edges if adj_edge != -1]
                for adj_edge in adj_edges:
                    self.assertEqual(to_location, id_to_edge[adj_edge][0])

        

    def test_make_edges(self):
        edges, adjs = make_edges(1)
        self.assertEqual(len(edges), 2*12+9)
        self.assertEqual(adjs[0], [(0,1), (0,3)])
        edges, adjs = make_edges(2)
        self.assertEqual(len(edges), 2*24+16)
        self.assertEqual(adjs[0], [(0,1), (0,4)])
        edges, adjs = make_edges(3)
        self.assertEqual(len(edges), 2*40+25)
        self.assertEqual(adjs[0], [(0,1), (0,5)])

    def test_compensate_edge(self):
        edges = compensate_edge((0,4), 1)
        self.assertEqual(edges[0], (0,1))
        self.assertEqual(edges[1], (1,4))

        edges = compensate_edge((0,8), 1)
        self.assertEqual(edges[0], (0,1))
        self.assertEqual(edges[1], (1,2))
        self.assertEqual(edges[2], (2,5))
        self.assertEqual(edges[3], (5,8))

if __name__ == '__main__':
    unittest.main()