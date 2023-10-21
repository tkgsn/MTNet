import unittest
from make_training_data import make_edges, write_edge_properties, convert, compliment_edge, run

class TestSample(unittest.TestCase):
    
    def setUp(self):
        # self.test_data_path = "/data/geolife/100/narrow_0_0_bin30_seed0/training_data.csv"
        self.test_data_path = "/data/geolife/1000/narrow_0_0_bin30_seed0/training_data.csv"

    def test_convert(self):
        # trajs = convert(self.test_data_path)
        # print(len(trajs))
        pass

    def test_write_edge_properties(self):
        pass

    def test_run(self):
        run(self.test_data_path)


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

    def test_compliment_edge(self):
        edges = compliment_edge((0,4), 1)
        self.assertEqual(edges[0], (0,1))
        self.assertEqual(edges[1], (1,4))

        edges = compliment_edge((0,8), 1)
        self.assertEqual(edges[0], (0,1))
        self.assertEqual(edges[1], (1,2))
        self.assertEqual(edges[2], (2,5))
        self.assertEqual(edges[3], (5,8))

if __name__ == '__main__':
    unittest.main()