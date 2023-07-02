from ex12 import *

x, t = get_data()
network = init_network()


batch_size = 1000

'''
#What is batch?

put a data in the network
X: 1x784 / W1: 784x50 / W2: 50x100 / W3: 100x10 => Y:1x10

put 100 datas in the network
-> NOT doing former procedures 100 times
-> just make the dimension of X larger
X: 100x784 /  W1: 784x50 / W2: 50x100 / W3: 100x10 => Y:100x10
'''

accuracy_cnt = 0


for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis = 1)

    '''
    #axis instruduction
    x = np.array([[0.1, 0.8, 0.1], 
                  [0.3, 0.1, 0.6], 
                  [0.2, 0.5, 0.3], 
                  [0.8, 0.1, 0.1]])
    y = np.argmax(x, axis=1)

    print(y) #[1 2 1 0]
    '''

    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt / len(x))))