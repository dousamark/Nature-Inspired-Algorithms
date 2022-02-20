from collections import namedtuple
import math
import functools
import numpy as np
import os
from xml.etree import ElementTree
import pprint

import matplotlib.pyplot as plt
from matplotlib import collections as mc

Vertex = namedtuple('Vertex', ['name', 'x', 'y', 'quantity'])

@functools.lru_cache(maxsize=None)
def distance(v1, v2):
    return math.sqrt((v1.x - v2.x)**2+(v1.y - v2.y)**2)

def fitness(vertices, distance, solution):
    solution_distance = 0
    for x, y in zip(solution, solution[1:]):
        solution_distance += distance(vertices[x], vertices[y])
    solution_distance += distance(vertices[solution[-1]], vertices[solution[0]])
    return solution_distance


def initialize_pheromone(N):
    return 0.01*np.ones(shape=(N,N))

def update_pheromone(pheromones_array, solutions, fits, Q=100, rho=0.6):
    pheromone_update = np.zeros(shape=pheromones_array.shape)
    for solution, fit in zip(solutions, fits):
        for x, y in zip(solution, solution[1:]):
            pheromone_update[x][y] += Q/fit
        pheromone_update[solution[-1]][solution[0]] += Q/fit
    
    return (1-rho)*pheromones_array + pheromone_update

def generate_solutions(vertices, pheromones, distance, N, alpha=1, beta=3):
    
    # pravdepodobnost vyberu dalsiho zakaznika
    def compute_prob(v1, v2):

        #pokud je jiz misto naplnene tak tam nebudu jezdit
        if (vertices[v2].quantity <= 0):
            return 0.000001
        
        dist = 1/distance(vertices[v1], vertices[v2])
        tau = pheromones[v1, v2]
        ret = pow(tau, alpha) * pow(dist,beta)
        return ret if ret > 0.000001 else 0.000001
            

    pheromones_shape = pheromones.shape[0]
    for i in range(N):
        available = list(range(pheromones_shape))
        solution = [np.random.randint(0, pheromones_shape)]
        available.remove(solution[0])

        restCapacity = maxCapacity
        #dokud jsou volne tak vybiram kam dal
        while available:
            #pokud mam jeste co rozvazet
            if (restCapacity > 0):
                probs = np.array(list(map(lambda x: compute_prob(solution[-1], x), available)))
                selected = np.random.choice(available, p=probs/sum(probs)) # vyber hrany

                #pokud moje kapacita staci na uspokojeni mnozstvi zakaznika tak pozadovane mnozstvi vylozim    
                if restCapacity > vertices[selected].quantity:
                    restCapacity -= vertices[selected].quantity
                
                #pokud ne tak nastavim zbytek kapacity na nulu, kterou pouzivam i jako indikator ze se nanasyti pozadavky zakaznika
                else:
                    restCapacity = 0

                available.remove(selected)
            else:
                #pokud jiz nemuzu nic nalozit nebo mam prazdno tak se vratim do depa a nalozim si plnou
                restCapacity = maxCapacity
                selected = 0

            solution.append(selected)
        yield solution

def ant_solver(vertices, distance, ants=10, max_iterations=300, alpha=1, beta=3, Q=100, rho=0.8):
    pheromones = initialize_pheromone(len(vertices))
    best_solution = None
    best_fitness = float('inf')
    
    for i in range(max_iterations):
        solutions = list(generate_solutions(vertices, pheromones, distance, ants,alpha=alpha, beta=beta))
        fits = list(map(lambda x: fitness(vertices, distance, x), solutions))
        pheromones = update_pheromone(pheromones, solutions, fits, Q=Q, rho=rho)
        
        for s, f in zip(solutions, fits):
            if f < best_fitness:
                best_fitness = f
                best_solution = s
        
        print(f'{i:4}, {np.min(fits):.4f}, {np.mean(fits):.4f}, {np.max(fits):.4f}')
    return best_solution, pheromones

vertices = []
# data_32.xml data_72.xml data_422.xml
osPath = os.path.abspath('data_422.xml')
dom = ElementTree.parse(osPath)

#globalni kapacita auta
maxCapacity = float(dom.find('fleet/vehicle_profile/capacity').text)

nodes = dom.findall('network/nodes/node')
depo = dom.find('network/nodes/node/[@type="0"]')
vertices.append(Vertex(int(depo.attrib['id']), float(depo.find('cx').text), float(depo.find('cy').text), 0))
for row in nodes:
    if (row.attrib['type'])=="1":
        row_id = row.attrib['id']
        row_quantity = dom.find("requests/request/[@node='{value}']".format(value=row_id))
        quantity = float(row_quantity.find('quantity').text)
        vertices.append(Vertex(int(row.attrib['id']), float(row.find('cx').text), float(row.find('cy').text), quantity))


pprint.pprint(vertices)

best_solution, pheromones = ant_solver(vertices, distance)


lines = []
colors = []
for i, v1 in enumerate(vertices):
    for j, v2 in enumerate(vertices):
        lines.append([(v1.x, v1.y), (v2.x, v2.y)])
        colors.append(pheromones[i][j])

lc = mc.LineCollection(lines, linewidths=np.array(colors))

plt.figure(figsize=(12, 8))
ax = plt.gca()
ax.add_collection(lc)
ax.autoscale()

solution = best_solution


print('Fitness: ', fitness(vertices, distance, solution))

solution_vertices = [vertices[i] for i in solution]
pprint.pprint(solution_vertices)

solution_lines = []
for i, j in zip(solution, solution[1:]):
    solution_lines.append([(vertices[i].x, vertices[i].y), (vertices[j].x, vertices[j].y)])
solution_lines.append([(vertices[solution[-1]].x, vertices[solution[-1]].y), (vertices[solution[0]].x, vertices[solution[0]].y)])
solutions_lc = mc.LineCollection(solution_lines, colors='red')
ax.add_collection(solutions_lc)
plt.show()