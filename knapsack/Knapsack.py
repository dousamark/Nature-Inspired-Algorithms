import numpy as np
import random

def getCostAndWeight(Individual):
    weight=0
    cost=0
    for i in range(len(Individual)):
        if(Individual[i]==1):
            weight+=items[i][1]
            cost+=items[i][0]
    return [cost,weight]

def cross(Individuals,fst,snd):
    #zacinam od jednicky a koncim o jednu driv aby se melo co krizit
    pointOfCrossing= random.randint(1, len(Individuals[0])-1)
    acc=np.copy(Individuals[fst][0:pointOfCrossing])

    for i in range(pointOfCrossing):
        Individuals[fst][i]= Individuals[snd][i]
    for i in range(pointOfCrossing):
        Individuals[snd][i]= acc[i]

    return Individuals

def flip(mutationBit):
    if(mutationBit==1):
        return 0
    else:
        return 1

def checkIfNotTooBig(Individuals):
    for i in range(len(Individuals)):
        [cost,weight]=getCostAndWeight(Individuals[i])
        if(weight>K):
            Individuals[i] = repairIndividual(Individuals[i],weight)
    return Individuals

def repairIndividual(Individual,weight):
    array=list(range(1, len(Individuals[0])-1))
    random.shuffle(array)
    for i in array:
        if(Individual[i]==1):
            Individual[i]=0
            weight-= items[i][1]
        if(weight<K):
            break
    return Individual

def decideFlip(mutationBit):
    Flip = random.uniform(0, 1)
    if(Flip>0.7):
        if(mutationBit==1):
            return 0
        else:
            return 1
    else:
        return mutationBit

parameters = input()

#nacteni parametru
N = int(parameters.split()[0])
K = int(parameters.split()[1])  

#nacteni items
items=[]

for i in range(N):
    item = input().split()
    items.append((int(item[0]), int(item[1])))

#chci mit NumberOfIndividuals jedincu jako charakteristicke vektory ke kazde polozce v items
NumberOfIndividuals= int(10*N)
Individuals=np.ndarray(shape=[NumberOfIndividuals,N])
for i in range(NumberOfIndividuals):
    Individuals[i] = np.random.choice([0,1],size=(N), p=[0.5,0.5])


fitness = [None] * NumberOfIndividuals
lenghtOfIndividual = len(Individuals[i])

#potom co vypustim testovat nahodne individuals se ujistim, ze vsechny budou v prvnim kole validni
checkIfNotTooBig(Individuals)

rounds=1
maxFitness=0
while(rounds<10000):
    #fitness
    for i in range(NumberOfIndividuals):
        [cost,weight]=getCostAndWeight(Individuals[i])
        if(weight>K):
            fitness[i]=0
        else:
            fitness[i]=cost

            if(fitness[i]>maxFitness):
                maxFitness=fitness[i]
                print()
                print("New best cost found:")
                print(maxFitness)

    #selekce
    newGen=np.ndarray(shape=[NumberOfIndividuals,N])
    for i in range(NumberOfIndividuals):
        newGen[i]=cross(random.choices(Individuals, weights=fitness, k=2),0,1)[0]
    Individuals=np.ndarray.copy(newGen)

    #krizeni
    crossing=0
    while(crossing+1<len(Individuals)):
        Individuals=cross(Individuals,crossing,crossing+1)
        crossing+=2 

    #mutace rate 20% ze se vybere a 30% ze se vybrany flipne 
    for i in range(NumberOfIndividuals):
        for j in range(len(Individuals[0])):
            shouldFlip = random.uniform(0, 1)
            if(shouldFlip>0.8):
                Individuals[i][j] = decideFlip(Individuals[i][j])

    rounds+=1
    
    if(rounds%5==0):
        checkIfNotTooBig(Individuals)
        

print("----------------")
print("Best Cost Overall:")
print(maxFitness)
"""
def getCostAndWeight(Individual):
    weight=0
    cost=0
    for i in range(len(Individual)):
        if(Individual[i]==1):
            weight+=items[i][1]
            cost+=items[i][0]
    return [cost,weight]

def cross(Individuals,fst,snd):
    #zacinam od jednicky a koncim o jednu driv aby se melo co krizit
    pointOfCrossing= random.randint(1, len(Individuals[0])-1)
    acc=np.copy(Individuals[fst][0:pointOfCrossing])

    for i in range(pointOfCrossing):
        Individuals[fst][i]= Individuals[snd][i]
    for i in range(pointOfCrossing):
        Individuals[snd][i]= acc[i]

    return Individuals

def flip(mutationBit):
    Flip = random.uniform(0, 1)
    if(Flip>0.5):
        if(mutationBit==1):
            return 0
        else:
            return 1
    else:
        return mutationBit

def checkIfNotTooBig(Individuals,tooMuch):
    for i in range(len(Individuals)):
        nbrOfOnes=countOnes(Individuals[i])
        if(nbrOfOnes>tooMuch):
            Individuals[i] = repairIndividual(Individuals[i],nbrOfOnes-tooMuch)
    return Individuals

def countOnes(Individual):
    counter =0
    for i in range(len(Individual)):
        if(Individual[i]==1):
            counter += 1
    return counter

def repairIndividual(Individual,toBeDeleted):
    now=True
    i=0
    #aby se kazda druha jednicka smazala
    while(toBeDeleted!=0):
        if(Individual[i]==1):
            if(now):
                toBeDeleted-=1
                Individual[i]=0
                now = False
            else:
                now = True
        i+=1
        if(i==lenghtOfIndividual):
            i=0
    return Individual
    
def allZeros(fitness):
    for i in range(len(fitness)):
        if(fitness[i]==1):
            return False
    return True


parameters = input()

#nacteni parametru
N = int(parameters.split()[0])
K = int(parameters.split()[1])  

#nacteni items
items=[]

for i in range(N):
    item = input().split()
    items.append((int(item[0]), int(item[1])))

#chci mit NumberOfIndividuals jedincu jako charakteristicke vektory ke kazde polozce v items
NumberOfIndividuals= int(10*N)
Individuals=np.ndarray(shape=[NumberOfIndividuals,N])
for i in range(NumberOfIndividuals):
    Individuals[i] = np.random.choice([0,1],size=(N), p=[0.9,0.1])


lenghtOfIndividual = len(Individuals[i])
checkIfNotTooBig(Individuals,int(N/5))
fitness = [None] * NumberOfIndividuals


rounds=1
maxFitness=0
while(rounds<10000):
    #fitness
    for i in range(NumberOfIndividuals):
        [cost,weight]=getCostAndWeight(Individuals[i])
        if(weight>K):
            fitness[i]=0
        else:
            fitness[i]=cost

            if(fitness[i]>maxFitness):
                maxFitness=fitness[i]
                print()
                print("New best cost found:")
                print(maxFitness)


    #selekce
    if(not allZeros(fitness)): 
        newGen=np.ndarray(shape=[NumberOfIndividuals,N])
        for i in range(NumberOfIndividuals):
            newGen[i]=cross(random.choices(Individuals, weights=fitness, k=2),0,1)[0]
        Individuals=np.ndarray.copy(newGen)

    #krizeni
    crossing=0
    while(crossing+1<len(Individuals)):
        Individuals=cross(Individuals,crossing,crossing+1)
        crossing+=2 

    #mutace rate 50%
    for i in range(NumberOfIndividuals):
        for j in range(len(Individuals[0])):
            shouldFlip = random.uniform(0, 1)
            if(shouldFlip>0.5):
                Individuals[i][j] = flip(Individuals[i][j])

    rounds+=1

    if (rounds%20 == 0):
        checkIfNotTooBig(Individuals,int(N/5))

print("----------------")
print("Best Cost Overall:")
print(maxFitness)
"""
