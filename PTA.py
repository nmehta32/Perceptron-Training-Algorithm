
import numpy as np
import matplotlib.pyplot as plt
w0 = np.random.uniform(-0.25,0.25)
w1 = np.random.uniform(-1,1)
w2 = np.random.uniform(-1,1)
n = int(input("Enter value of N (ie number of elements)"))
xi = np.random.uniform(-1,1,size = (n,2))
b = np.ones((n,1))
xi_ = np.hstack((b , xi))

w = [w0 , w1 , w2]
print(w, "Perfect Weights For classification")
S0 = np.array([])
S1 = np.array([])
s0 = []
s1= []
for i in range(n):
  q = np.dot(w, xi_[i])

  if q < 0:
    S0 = np.vstack(xi_[i])
    s0.append(S0)

  else:
    S1 = np.vstack(xi_[i])
    s1.append(S1)

dataset = s0 + s1
s0 = np.asarray(s0)
s1 = np.asarray(s1)
dataset = np.asarray(dataset)

def plot (w,s0,s1):
  x1 = np.linspace(-5,5,n)
  x2 = (-w[0]-w[1]*x1)/w2
  plt.plot(x1, x2, '-b', label='perfect classification line')
  plt.title('Graph of w0 + w1x1 + w2x2 = 0')
  plt.xlabel('x1', color='#1C2833')
  plt.ylabel('x2', color='#1C2833')
  plt.legend(loc='upper left')
  S0_x = []
  S0_y = []
  S1_x= []
  S1_y= []
  for i in range(len(s0)):
    S0_x.append(s0[i][1])
    S0_y.append(s0[i][2])
  for i in range(len(s1)):
    S1_x.append(s1[i][1])
    S1_y.append(s1[i][2])
  scat1 = plt.scatter(S0_x,S0_y,color = 'red', label='Elements where the output of network is 0')
  scat2 = plt.scatter(S1_x,S1_y,color = 'green', label = 'Elements where the output of network is 1')
  plt.legend((scat1, scat2),
           ('Elements where the output of network is 0', 'Elements where the output of network is 1'),
           scatterpoints=1,
           loc='best',
           ncol=3,
           fontsize=8)
  plt.ylim([-1.5,1.5])
  plt.xlim([-1.5,1.5])
  plt.grid()
  plt.show() 



w0_ = np.random.uniform(-1,1)
w1_ = np.random.uniform(-1,1)
w2_ = np.random.uniform(-1,1)
w_ = [w0_ , w1_ , w2_]    
print(w_, 'randomly selected weights')
##notepad

lab1 = np.ones(len(s0),dtype = int)
lab2 = np.zeros(len(s1),dtype = int)

label = np.append(lab1,lab2)

def miscal(dataset, w_):
  miscal_ = 0
  for i in range(len(dataset)):
    y = w_[0] + (dataset[i][1]*w_[1]) + (dataset[i][2]*w_[2])
    if (y[0] >= 0):
      y[0] = int(1)
    else:
      y[0] = int(0)
    if (y[0] != label[i].astype(int)):
      miscal_ = miscal_ +1
  return miscal_


def PTA(w_,xi):
  epoch = 0
  omegas = []
  miss = []
  while (miscal(dataset, w_) != 0):
        miss.append(miscal(dataset, w_))
        print ('Number of missclassifications: ', miss[epoch])
        epoch = epoch + 1
        print ('Epoch Number: ', epoch)
        
        for i in range(len(dataset)):
            y = w_[0] + (dataset[i][1]*w_[1]) + (dataset[i][2]*w_[2]) ##dot product
            if y[0] >= 0:
              y = 1
            else:
              y = 0

            difference = label[i]-y
            updated_input =[1]+dataset[i][0:2]

            if difference != 0:
              ip1 = dataset[i][0]*eta*difference
              ip2 = dataset[i][1]*eta*difference
              ip3 = dataset[i][2]*eta*difference
              a = w_[0]+ip1
              b = w_[1]+ip2
              c = w_[2]+ip3
              w_ = [a, b, c]
              w_ = np.asarray(w_)
        #print ('Updated weights: ', w_)
        omegas.append(w_)
  final_misclassification = miscal(dataset,w_)
  print ('Final weights: ', w_)
  return omegas, miss


plot(w,s0,s1)

eta = 1
PTA(w_,dataset)
print ('Initial weight: ' , w_)
omegas=[]
omegas, miss = PTA(w_,xi_)
n_epochs = range(len(omegas)+1)
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(n_epochs, miss+[0], c = 'blue')

plt.title('Misclassification vs Epochs with ETA =' + str(eta))
plt.ylabel('Number of Misclassifications')
plt.xlabel('Number of Epochs')
plt.show()


eta = 0.1
PTA(w_,dataset)
print ('Initial weight: ' , w_)
omegas=[]
omegas, miss = PTA(w_,xi_)
n_epochs = range(len(omegas)+1)
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(n_epochs, miss+[0], c = 'blue')

plt.title('Misclassification vs Epochs with ETA =' + str(eta))
plt.ylabel('Number of Misclassifications')
plt.xlabel('Number of Epochs')
plt.show()


eta = 10
PTA(w_,dataset)
print ('Initial weight: ' , w_)
omegas=[]
omegas, miss = PTA(w_,xi_)
n_epochs = range(len(omegas)+1)
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(n_epochs, miss+[0], c = 'blue')

plt.title('Misclassification vs Epochs with ETA =' + str(eta))
plt.ylabel('Number of Misclassifications')
plt.xlabel('Number of Epochs')
plt.show()
