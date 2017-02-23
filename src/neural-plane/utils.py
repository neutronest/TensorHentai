#-*- coding:utf-8 -*-
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# return area [-1, 1]
def random_clamped():
    return random.random()*2-1
