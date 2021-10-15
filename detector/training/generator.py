#Генератор для создания случайных графиков и диаграмм, используемых для обучения нейросети.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import random
import time
import os
import string
import itertools
import networkx as nx
import numpy.random as rnd
import sys
sys.path.append(".")

#Количество графиков/диаграмм, которое нужно сохранить для обучения.
count = 10
#Максимум и минимум точек на графиках
high = 30
low = 2
path = '../science'
os.makedirs(path, exist_ok=True)

def generate_graphs():
    for i in range(count):
        #случайный выбор, будет график убывающим, возрастающим или случайным.
        howsort = random.choice(['asc', 'desc', 'none'])
        #случайное количество точек на графике
        co = random.randint(low, high)
        #генерация массива случайных координат
        x = np.random.randint(0, 1000, co)
        y = np.random.randint(0, 1000, co)
        #случайный выбор, будет ли сортировка координат по ОХ
        if (random.randint(1,100) > 20):
            x.sort()
        #сортировка координат по OY в соответствии с выбором
        if (howsort == 'asc'):
            y.sort()
        elif (howsort == 'desc'):
            y = sorted(y, reverse=True)
        #создание и сохранение графика
        plt.plot(x, y)
        name = 'graph' + str(int(time.time() * 1000))
        plt.savefig(path + '/' + name + '.png')
        #случайный выбор, завершить рисование графика или добавить ещё один из следующего цикла
        if (random.randint(1,100) > 40):
            plt.close()

def generate_diagram():
    for i in range(count):
        #случайный выбор типа диаграммы
        dtype = random.choice(['hist', 'bar', 'barh', 'circle'])
        #диаграммы строятся на основе сгенерированного рандомного набора данных
        if (dtype == 'hist'):
            rng = np.random.RandomState(random.randint(1, 1000)) 
            a = np.hstack((rng.normal(size=1000), rng.normal(loc=random.randint(1, 10), scale=random.randint(1, 5), size=1000)))
            plt.hist(a, bins='auto')
            name = 'histo' + str(int(time.time() * 1000))
            plt.savefig(path + '/' + name + '.png')
        elif (dtype == 'bar' or dtype == 'barh'):
            groups = [f"G{i}" for i in range(low, 15)]
            counts = np.random.randint(1, 100, len(groups))
            if (dtype == 'bar'):
                plt.bar(groups, counts)
                name = 'bar' + str(int(time.time() * 1000))
            elif (dtype == 'barh'):
                plt.barh(groups, counts)
                name = 'barh' + str(int(time.time() * 1000))
            plt.savefig(path + '/' + name + '.png')
        elif (dtype == 'circle'):
            co = random.randint(low, 15)
            letters = np.array(list(string.ascii_lowercase))
            values = np.random.randint(0, 100, co)
            groups = [''.join(np.random.choice(letters, size=5)) for i in range(co)]
            plt.pie(values, labels=groups)
            name = 'circle' + str(int(time.time() * 1000))
            plt.savefig(path + '/' + name + '.png')
        plt.close()

def generate_schema():
        #в качестве схемы используется граф Эрдьёша-Реньи на основании количества узлов и вероятности их попарного соединения.
        for i in range(count):
            schema = nx.Graph()
            letters = np.array(list(string.ascii_lowercase))
            nodes = [''.join(np.random.choice(letters, size=3)) for i in range(4, 15)]
            schema.add_nodes_from(nodes)
            for pair in itertools.permutations(nodes, 2):
                try:
                    if rnd.random() < 0.2: schema.add_edge(*pair)
                except:
                    print("Graph construction error")
            f = plt.figure()
            #Случайным образом выбирается форма узлов графа.
            form = random.choice(['s', 'd', 'o' ,'h', 'v'])
            axf = f.add_subplot(111)
            nx.draw(schema, ax=axf, node_shape=form, node_size=random.randint(1000, 1300), with_labels=True)
            axf.margins(0.3)
            name = 'schema' + str(int(time.time() * 1000))
            f.savefig(path + '/' + name + '.png')
            plt.close()
            

generate_graphs()
generate_diagram()
generate_schema()