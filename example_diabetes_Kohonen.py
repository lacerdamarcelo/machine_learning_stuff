from minisom import MiniSom
from numpy import genfromtxt,array,linalg,zeros,mean,std,apply_along_axis
import matplotlib.pyplot as plt

"""
    This script shows how to use MiniSom on the Diabetes dataset.
    In partucular it shows how to train MiniSom and how to visualize the result.
    ATTENTION: pylab is required for the visualization.        
"""

data = genfromtxt('diabetes.csv', delimiter=',',usecols=(0,1,2,3,4,5,6,7))
data = apply_along_axis(lambda x: x/linalg.norm(x),1,data) # data normalization

### Initialization and training ###
#number of neurons in X and Y, number of attributes, sigma and learning_rate
som = MiniSom(15,15,8,sigma=1.0,learning_rate=0.5)
#som.random_weights_init(data)
print("Training...")
#data and number of iterations
som.train_random(data,10000) # random training
print("\n...ready!")

### Plotting the response for each pattern in the boston dataset ###
from pylab import plot,axis,show,pcolor,colorbar,bone
bone()
pcolor(som.distance_map().T) # plotting the distance map as background
colorbar()
target = genfromtxt('diabetes.csv',delimiter=',',usecols=(8),dtype=str) # loading the labels
t = zeros(len(target),dtype=int)
t[target == '0'] = 0
t[target == '1'] = 1
# use different colors and markers for each label
markers = ['o','s']
colors = ['r','g']


for cnt,xx in enumerate(data):
	w = som.winner(xx) # getting the winner
	# palce a marker on the winning position for the sample xx
	plot(w[0]+.5,w[1]+.5,markers[t[cnt]],markerfacecolor='None',markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=2)
	axis([0,som.weights.shape[0],0,som.weights.shape[1]])
	show() # show the figure

plt.savefig("teste")
