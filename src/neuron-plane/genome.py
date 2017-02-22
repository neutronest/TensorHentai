#-*- coding:utf-8 -*-

class Genome():
    def __init__(self, score_, network_weights_):
        self.score = score_
        self.network_weights = network_weights_
        return

class Generation():
    def __init__(self, SCORE_SORT):
        self.genomes = []
        self.score_sort = SCORE_SORT
        return

    def add_genome(self, genome):
        i = 0
        for i in xrange(len(self.genomes)):
            if self.score_sort > self.

