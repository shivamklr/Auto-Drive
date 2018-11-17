import numpy as np
import matplotlib.pyplot as plt
def draw(x1,x2):
	'''This function takes two coordinates and draws a line '''
	ln = plt.plot(x1,x2)

def sigmoid	(score):
	'''This function takes in an array and retuns the sigmoid of each individual element in that array'''
	return 1/(1+np.exp(-score))

n_pts=10
#number of desired vertices 

np.random.seed(0)
bias=np.ones(n_pts)
top_region=np.array([np.random.normal(10,2,n_pts), np.random.normal(12,2,n_pts),bias]).T
bottom_region= np.array([np.random.normal(5,2, n_pts), np.random.normal(6,2, n_pts),bias]).T
all_points= np.vstack((top_region,bottom_region))
w1= -0.2
w2= -0.35
b= 3.5
line_parameters=np.matrix([w1,w2,b]).T
#forming an array of the weights that effect the line 

x1 = np.array([bottom_region[:,0].min(), top_region[:,0].max()])
x2 = -b/w2 + (x1*(-w1/w2))
#taking initial coordinates for the line

line_combo = all_points * line_parameters
print(line_combo)
# matrix multiplication {of (100,3) & (3,1) matrices}
# line_combo is matrix that has error sum for individual vertex for all 100 vertices

probabilities=sigmoid(line_combo)
# print(probabilities)
_, ax= plt.subplots(figsize=(4,4))
ax.scatter(top_region[:,0], top_region[:,1], color='r')
ax.scatter(bottom_region[:,0], bottom_region[:,1], color='b')
draw(x1,x2)
plt.show()