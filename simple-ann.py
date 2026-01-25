import math

x = [
    [1.0, 0.1, -0.2],
    [1.0, -0.1, 0.9],
    [1.0, 1.2, 0.1],
    [1.0, 1.1, 1.5],
]

t = [0, 0, 0, 1]

w = [1, -1, 1]

def fp_ANN(x,w,t):
    total_e = 0
    e = []
    y = []
    
    for n in range(len(x)):
        v = 0
        
        for d in range(len(x[0])):
            v += x[n][d] * w[d]
        
        y.append(1/(1+math.e**(-v))) # Sigmoid
        e.append(-t[n]*math.log(y[n]) - (1-t[n])*math.log(1-y[n])) # BCE
       
    total_e = sum(e)/len(x)
    return (y, w, total_e)
        
# y, w, total_e = fp_ANN(x,w,t)
# print(y)
# print(w)
# print(total_e)

def fp_bp_ANN(x, w, t, iter, lr):
    total_e = 0
    
    for i in range(iter):
        e = []
        y = []
        
        for n in range(len(x)): # over all len(x) data points
            v = 0
            
            for d in range(len(x[0])): # over all components of a single data point
                v += x[n][d] * w[d]

            y.append(1/(1+math.e**(-v)))
            e.append(-t[n]*math.log(y[n]) - (1-t[n])*math.log(1-y[n])) # BCE
                
            for p in range(len(w)): # update all weights for each data point in x
                d = 2*x[n][p]*(y[n]-t[n])*(1-y[n])*y[n]
                w[p] -= lr*d
    
    total_e = sum(e)/len(x)
    return (y, x, total_e)

y, w, total_e = fp_bp_ANN(x, w, t, 100, 0.05)
print(y)
print(w)
print(total_e)