from nnet import NeuralNetwork

print "Verifying gradients with sigmoid activation"
m = NeuralNetwork()
m.L2 = 0
m.L1 = 0
m.tanh = False
m.verify_gradients()

print ""
print "Verifying gradients with tanh activation"
m.tanh = True
m.verify_gradients()

print ""
print "Verifying gradients with L2 regularization"
m.L2 = 0.001
m.verify_gradients()

print ""
print "Verifying gradients with L1 regularization"
m.L2 = 0
m.L1 = 0.001
m.verify_gradients()

