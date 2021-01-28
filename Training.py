#https://github.com/CodeReclaimers/neat-python
#https://gitlab.com/lucasrthompson/Sonic-Bot-In-OpenAI-and-NEAT/-/blob/master/playback.py


import gym
import numpy as np
import neat
import pickle
import time

env = gym.make('FrozenLake8x8-v0')


resume = False
restore_file = "neat-checkpoint-601"
runs_per_net = 5

def gen(x):
    #Render Field into array
    if x == "b'S'":
        return 0.1
    if x == "b'F'":
        return 0.5
    if x == "b'H'":
        return 0
    if x == "b'G'":
        return 1


def eval_genomes(genomes, config):

    times = []

    for genome_id, genome in genomes:

        fitnesses = []


        for runs in range(runs_per_net):

            start=time.time()
            ob = env.reset()
            ac = env.action_space.sample()


            net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

            current_max_fitness = 0
            fitness_current = 0
            counter = 0
            done = False

            while not done:

                #env.render()

                field =np.ndarray.flatten(env.desc)
                field = [str(x) for x in field]

                imgarray = np.asarray(list(map(gen,field)))

                imgarray[ob] = 0.8 # Derzeitige Position

                nnOutput = net.activate(imgarray)


                ob, rew, done, info = env.step(np.argmax(nnOutput))


                if rew > 0: # Fals goal erreicht wurde
                        fitness_current += 10000000
                        done = True


                #if done==False:
                #    rew = rew + 0.1 #Reward fürs überleben

                fitness_current += rew

                if fitness_current > current_max_fitness:
                    current_max_fitness = fitness_current
                    counter = 0
                else:
                    counter += 1

                if done or counter == 250:
                    done = True
                    #print(genome_id, fitness_current)
                    fitnesses.append(fitness_current)
                    times.append(time.time()-start)



        genome.fitness = min(fitnesses)

    print('Avarge SimTime: ', np.mean(times))
    
    

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

if resume == True:
    p = neat.Checkpointer.restore_checkpoint(restore_file)
else:
    p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(1))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)





ob = env.reset()
ac = env.action_space.sample()
net = neat.nn.recurrent.RecurrentNetwork.create(winner, config)

current_max_fitness = 0
fitness_current = 0
counter = 0
done = False

while not done:

    #env.render()

    field = np.ndarray.flatten(env.desc)
    field = [str(x) for x in field]


    def gen(x):
        if x == "b'S'":
            return 0.1
        if x == "b'F'":
            return 0.5
        if x == "b'H'":
            return 0
        if x == "b'G'":
            return 1


    imgarray = np.asarray(list(map(gen, field)))

    imgarray[ob] = 0.8  # Derzeitige Position

    nnOutput = net.activate(imgarray)

    ob, rew, done, info = env.step(np.argmax(nnOutput))

    if rew > 0:  # Fals goal erreicht wurde
        fitness_current += 10000000
        done = True


    # if done==False:
    #    rew = rew + 0.1 #Reward fürs überleben

    fitness_current += rew

    if fitness_current > current_max_fitness:
        current_max_fitness = fitness_current
        counter = 0
    else:
        counter += 1

    if done or counter == 250:
        done = True
        print('Done: ', True)
        print('Counter: ', counter)




import Neat_Viz
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

Neat_Viz.plot_stats(stats, filename='avg_fitness.png')
Neat_Viz.plot_species(stats, view=True, filename="ctrnn-speciation.png")
Neat_Viz.draw_net(config, winner, True,fmt='png')