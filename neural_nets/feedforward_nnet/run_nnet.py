import numpy as np
import os
import sys
import fcntl
import copy
from string import Template
import mlpython.datasets.store as dataset_store
import mlpython.mlproblems.generic as mlpb
from nnet import NeuralNetwork

sys.argv.pop(0);	# Remove first argument

# Check if every option(s) from parent's script are here.
if 7 != len(sys.argv):
    print "Usage: python run_nnet.py lr dc sizes L2 L1 seed tanh"
    print ""
    print "Ex.: python run_nnet.py 0.1 0 [20,10] 0 0 1234 False"
    sys.exit()

# Set the constructor
str_ParamOption = "lr=" + sys.argv[0] + ", " + "dc=" + sys.argv[1] + ", " + "sizes=" + sys.argv[2] + ", " + "L2=" + sys.argv[3] + ", " + "L1=" + sys.argv[4] + ", " + "seed=" + sys.argv[5] + ", " + "tanh=" + sys.argv[6]
str_ParamOptionValue = sys.argv[0] + "\t" + sys.argv[1] + "\t" + sys.argv[2] + "\t" + sys.argv[3] + "\t" + sys.argv[4] + "\t" + sys.argv[5] + "\t" + sys.argv[6]
try:
    objectString = 'myObject = NeuralNetwork(n_epochs=1, ' + str_ParamOption + ')'
    exec objectString
    #code = compile(objectString, '<string>', 'exec')
    #exec code
except Exception as inst:
    print "Error while instantiating NeuralNetwork (required hyper-parameters are probably missing)"
    print inst

print "Loading dataset..."
trainset,validset,testset = dataset_store.get_classification_problem('ocr_letters')

print "Training..."
# Early stopping code
best_val_error = np.inf
best_it = 0
str_header = 'best_it\t'
look_ahead = 5
n_incr_error = 0
for stage in range(1,500+1,1):
    if not n_incr_error < look_ahead:
        break
    myObject.n_epochs = stage
    myObject.train(trainset)
    n_incr_error += 1
    outputs, costs = myObject.test(trainset)
    errors = np.mean(costs,axis=0)
    print 'Epoch',stage,'|',
    print 'Training errors: classif=' + '%.3f'%errors[0]+',', 'NLL='+'%.3f'%errors[1] + ' |',
    outputs, costs = myObject.test(validset)
    errors = np.mean(costs,axis=0)
    print 'Validation errors: classif=' + '%.3f'%errors[0]+',', 'NLL='+'%.3f'%errors[1]
    error = errors[0]
    if error < best_val_error:
        best_val_error = error
        best_it = stage
        n_incr_error = 0
        best_model = copy.deepcopy(myObject)

outputs_tr,costs_tr = best_model.test(trainset)
columnCount = len(costs_tr.__iter__().next())
outputs_v,costs_v = best_model.test(validset)
outputs_t,costs_t = best_model.test(testset)

# Preparing result line
str_modelinfo = str(best_it) + '\t'
train = ""
valid = ""
test = ""
# Get average of each costs
for index in range(columnCount):
    train = str(np.mean(costs_tr,axis=0)[index])
    valid = str(np.mean(costs_v,axis=0)[index])
    test = str(np.mean(costs_t,axis=0)[index])
    str_header += 'train' + str(index+1) + '\tvalid' + str(index+1) + '\ttest' + str(index+1)
    str_modelinfo += train + '\t' + valid + '\t' + test
    if ((index+1) < columnCount): # If not the last
        str_header += '\t'
        str_modelinfo += '\t'
str_header += '\n'
result_file = 'results_nnet_ocr_letters.txt'

# Preparing result file
header_line = ""
header_line += 'lr\tdc\tsizes\tL2\tL1\tseed\ttanh\t'
header_line += str_header
if not os.path.exists(result_file):
    f = open(result_file, 'w')
    f.write(header_line)
    f.close()

# Look if there is optional values to display
if str_ParamOptionValue == "":
    model_info = [str_modelinfo]
else:
    model_info = [str_ParamOptionValue,str_modelinfo]

line = '\t'.join(model_info)+'\n'
f = open(result_file, "a")
fcntl.flock(f.fileno(), fcntl.LOCK_EX)
f.write(line)
f.close() # unlocks the file

