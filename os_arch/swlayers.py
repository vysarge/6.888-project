import numpy as np
from scipy.signal import correlate2d

def ReLU(ifmap):
	(row, col, chn) = ifmap.shape
	for ch in range(chn):
		for r in range(row):
			for c in range(col):
				if ifmap[r][c][ch] <= 0:
					ifmap[r][c][ch] = 0

	return ifmap



def MAXPOOL(ifmap, pool_size):
	(row, col, chn) = ifmap.shape
	assert(row%pool_size == 0 and col%pool_size == 0)
	out_r = row//pool_size
	out_c = col//pool_size
	ofmap = np.zeros([out_r, out_c, chn])#.astype(np.int64)
	for ch in range(chn):
		for o_r in range(out_r):
			for o_c in range(out_c):
				m = max(ifmap[i][j][ch] for i in range(o_r*pool_size,o_r*pool_size+pool_size)for j in range(o_c*pool_size,o_c*pool_size+pool_size))
				#m = int(max(ifmap[i][j][ch] for i in range(o_r*pool_size,o_r*pool_size+pool_size)for j in range(o_c*pool_size,o_c*pool_size+pool_size)))
				#print(m)
				ofmap[o_r][o_c][ch] = m
				#print(ofmap[o_r][o_c][ch])
	return ofmap

# drawn from stimulus.py
def conv(x, W, b):
    # print x.shape, W.shape, b.shape
    y = np.zeros([x.shape[0], x.shape[1], W.shape[3]])#.astype(np.int64)
    for out_channel in range(W.shape[3]):
        for in_channel in range(W.shape[2]):
            W_c = W[:, :, in_channel, out_channel]
            x_c = x[:, :, in_channel]
            y[:, :, out_channel] += correlate2d(x_c, W_c, mode="same")
        y[:, :, out_channel] += b[out_channel]
    return y

# computes softmax of input list
def softmax(x):
	expx = np.exp(x)
	norm = np.sum(expx)
	soft = expx/norm
	return soft


#ifmap = np.reshape(np.arange(4*4*4), (4,4,4))

#relu_out = ReLU(ifmap)
#pool_out = MAXPOOL(ifmap, 2)

#print("-----------pool_out--------------")
#print(pool_out)
