'''
Tutorial for learning Q-Learning und OpenAi Gym

Look for futher instructions :https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2
'''
import matplotlib.pyplot as plt
import gym
import numpy as np

#Load Gym and Q-Table Structure

env = gym.make('FrozenLake8x8-v0')

Q = np.zeros([env.observation_space.n,env.action_space.n])
'''Actions:
0 - left
1 - down
2 - right
3 - up
'''

eta = 0.628
gma = .9
epis = 10000
rev_list = [] #rewards per epsidode
winningrate = []
winningrate100= [] # winningrate every Hundert steps
# Q-Learning Algorithm
i100 = 0

for i in range(epis):
    s = env.reset() # Ergibt State = 0 bzw s = 0
    rAll = 0 # Alle Rewards in einer Session
    d = False
    j = 0
    actions = []
    i100 += 1

    #Learning
    while j < 99:
        #env.render()
        j+=1

        #Choose Action from Q-Table
        # Erzeugt für jede Instanz ein Array aus einer Normalverteilung und addiert sie auf Q[s,:] auf
        # je höher i, desto geringer wird der Einfluss der Zufalls Variablen
        # i = 1: Q[s,:] + array([[-0.50345129,  0.10723116, -0.59345065, -0.30912405]])
        # i = 100: Q[s,:] + array([[-0.00017393, -0.00727869, -0.01084878, 0.02232659]])
        # a ist das Maximum aus Array

        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1/(i+1)))

        actions.append(a)

        #Get new Stat & reward from env
        s1,r,d,_ = env.step(a)

        #Update Q Table
        #Formel wurde nach eta umgestellt. In der ersten Iteration wird Q[s1,:] =[0,0,0,0] sein
        # In den ersten Iterationen wird r eine besondere Rolle spielen, da r den Q-Table verändern wird
        Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])

        rAll += r
        s = s1

        if r > 0:
            #renderfunktion
            #env.render()
            #print(actions)
            pass

        if d == True:
            break



    rev_list.append(rAll)
    winningrate.append(sum(rev_list) / epis)


    if i100 == 100:
        winningrate100.append(sum(rev_list[i-100:i]) / i100)
        i100 = 0
        print('calculated ' + str(i+1) + ' Episodes...')
    #env.render()

print("Reward Sum on all episodes " + str(sum(rev_list)/epis))
print("Final Values Q-Table")
print(Q)

#Fortschritt des Algorithmus
#plt.plot(np.cumsum(np.asarray(rev_list)))
#plt.show()

#plt.plot(winningrate)
#plt.show()

plt.plot(winningrate100) # Winning Rate over hundert games
plt.show()