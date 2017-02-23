#-*- coding:utf-8 -*-
import random
import utils
class Genome(object):
    # the neuron weight is the genome during generation
    def __init__(self, score_, network_weights_):
        self.score = score_
        self.network_weights = network_weights_
        return

class Generation(object):
    def __init__(self,
                 SCORE_SORT,
                 MUTATION_RATE,
                 ELITISM,
                 RANDOM_BEHAVIOR,
                 POPULATION,
                 N_CHILD):
        self.genomes = []
        self.score_sort = SCORE_SORT
        self.mutation_rate = MUTATION_RATE
        self.elitism = ELITISM
        self.population = POPULATION
        self.random_behavior = RANDOM_BEHAVIOR
        self.n_chlid = N_CHILD

        return

    def add_genome(self, genome):
        i = 0
        for i in xrange(len(self.genomes)):
            # just insert by sort
            if self.score_sort < 0:
                if genome.score > self.genomes[i].score:
                    break
            else:
                if genome.score < self.genomes[i].score:
                    break
        self.genomes.insert(i, genome)
        return

    def breed(self, genome1, genome2, n_child):
        #
        next_genomes = []
        for t in xrange(0, n_child):

            # deep copy from genome1
            cur_genome = Genome(0,
                                {"weights": genome1.network_weights["weights"][:],
                                 "network": genome1.network_weights["network"][:]})
            for i in xrange(len(genome1.network_weights)):
                if random.random() <= 0.5:
                    cur_genome.network_weights["weights"][i] = genome2.network_weights["weights"][i]
            for i in xrange(len(cur_genome.network_weights)):
                if random.random() >= self.mutation_rate:
                    cur_genome.network_weights["weights"][i] += random.random()
            next_genomes.append(cur_genome)
        return next_genomes

    def next_generation(self):
        next_weights = []
        for i in xrange(0, round(len(self.elitism * self.population))):
            if len(next_weights) < self.population:
                next_weights.append(self.genomes[i].network_weights)
        for i in xrange(0, round(len(self.random_behavior * self.populatino))):
            # TODO: check this logic
            n = self.genomes[i].network_weights
            for k in xrange(0, len(n["weights"])):
                n["weights"][k] = utils.random_clamped()
            if len(next_weights) < population:
                next_weights.append(n)
        max_n = 0
        while True:
            for i in xrange(0, max_n):
                genome_childs = self.breed(self.genomes[i],
                                           self.genomes[max_n],
                                           self.n_child)
                for j in xrange(0, len(genome_childs)):
                    next_weights.append(genome_childs[j].network_weights)
                    if len(next_weights) > self.population:
                        return next_weights
            max_n += 1
            if max_n >= len(self.genomes)-1:
                max_n = 0


class NeuronEvolution(object):

    def __init__(self, POPUPLATION):
        self.generations = []
        self.population = POPULATION

    def restart(self):
        self.__init__()

    def get_next_generations(self):
        networks = []
        if len(self.generations) == 0:
            # first generation
            out = []
            for i in xrange(0, len())
