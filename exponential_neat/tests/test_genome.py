from genetic_algorithm.genome import Genome
import pandas as pd

class TestGenome:
    G = Genome(3)

    def test_create(self):
        net = self.G.newNet(random_weights=False)
        edges = list(net.edges())
        assert list(net.edges()) == [(0, 3), (1, 3), (2, 3)]
        
    def test_evaluate(self):
        pass
    
    def test_crossover(self):
        pass
    
    def test_new_node(self):
        pass
    
    def test_new_connection(self):
        pass
    

