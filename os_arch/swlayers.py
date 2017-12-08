import numpy as np

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
	ofmap = np.zeros([out_r, out_c, chn]).astype(np.int64)
	for ch in range(chn):
		for o_r in range(out_r):
			for o_c in range(out_c):
				m = int(max(ifmap[i][j][ch] for i in range(o_r*pool_size,o_r*pool_size+pool_size)for j in range(o_c*pool_size,o_c*pool_size+pool_size)))
				#print(m)
				ofmap[o_r][o_c][ch] = m
				#print(ofmap[o_r][o_c][ch])
	return ofmap



ifmap = np.reshape(np.arange(4*4*4), (4,4,4))

relu_out = ReLU(ifmap)
pool_out = MAXPOOL(ifmap, 2)

print("-----------pool_out--------------")
print(pool_out)
