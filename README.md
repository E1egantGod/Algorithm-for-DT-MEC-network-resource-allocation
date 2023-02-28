1.Data generator: These files generate data sets for training and testing according to the given parameter range.

getDate_conv.py

getDate_fc.py

getDate_pool.py

2.DNN Layer Latency Predictor: These files predict the inference delay of the DNN layer according to the given DNN layer parameters.

train_test_conv.py

train_test_fc.py

train_test_pool.py

3.DNN Partition Module: These files find the best partition point of DNN by searching for min st-cut.

MinCut.py

ford_fulkerson.py

4.RL Agent: These files use the A3C algorithm to manage the communication resource allocation (such as transmission power) of the end device.

CMA3C.py

utils.py
