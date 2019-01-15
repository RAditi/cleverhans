import os 

start = 201;
end = 201;

epsilon = 0.1;
num_classes = 10;

size_matrix = 1135;

Command = "matlab_r2016b -r \" multi_layer(4, " + str(size_matrix) + ", " +str(epsilon) + "," + str(num_classes) + "," + str(start) + "," + str(end) + " )\""
os.system(Command)


# SDP_value = sio.loadmat('SDP_result.mat')
# SDP_value = SDP_value['val']

# SDP_matrix = sio.loadmat('SDP_matrix.mat')
# SDP_matrix = SDP_matrix['M']


# np.save("Matrix" + str(i) , SDP_matrix)
# np.save("Value" + str(i) , SDP_value)
    
