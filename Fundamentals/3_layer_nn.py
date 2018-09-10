import numpy as np

def sigmoid(val, der = False):
	if(der):
		return val * (1 - val)
	else:
		return 1 / (1 + np.exp(-val))
	
def main():
	x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	y = np.array([[0, 1, 1, 0]]).T
	
	np.random.seed(1)
	
	syn0 = 2 * np.random.random((2, 4)) - 1
	syn1 = 2 * np.random.random((4, 1)) - 1
	
	for _ in range(80000):
		
		l0 = x
		l1 = sigmoid(np.dot(l0, syn0))
		l2 = sigmoid(np.dot(l1, syn1))
		
		l2_err = y - l2
		
		l2_del = l2_err * sigmoid(l2, True)
		l1_del = sigmoid(l1, True) * np.dot(l2_del, syn1.T)
		
		syn0 += l0.T.dot(l1_del)
		syn1 += l1.T.dot(l2_del)
		
		if _ % 5000 == 0:
			print('l1:')
			print(l1)
			print('l2:')
			print(l2)
			print('Error:')
			print(np.mean(np.abs(l2_err)))
	
	print('Weights0:')
	print(syn0)
	print('Weights1:')
	print(syn1)
	l0 = x
	l1 = sigmoid(np.dot(l0, syn0))
	l2 = sigmoid(np.dot(l1, syn1))
	print('l2:')
	print(l2)

if __name__ == '__main__':
	main()
