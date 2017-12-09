from scipy.signal import correlate2d
import numpy as np

from nnsim.module import Module
from serdes import InputSerializer, OutputDeserializer

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

class Stimulus(Module):
    def instantiate(self, arr_y, block_size, num_nonzero, input_chn, output_chn, pruner_name):
        # PE static configuration (immutable)
        self.arr_y = arr_y
        self.block_size = block_size
        self.num_nonzero = num_nonzero

        self.input_chn = input_chn
        self.output_chn = output_chn

        self.serializer = InputSerializer(self.input_chn, 
            self.arr_y, self.block_size, self.num_nonzero, pruner_name)
        self.deserializer = OutputDeserializer(self.output_chn, 
            self.arr_y, self.block_size, self.num_nonzero)
        self.ofmap = None

    def configure(self, image_size, filter_size, in_chn, out_chn, ifmap, weights, bias, debug, keep_max, do_premature_prune):
        # Test data
        #ifmap = np.zeros((image_size[0], image_size[1],
        #    in_chn)).astype(np.int64)

        # if (debug): # Test values that should be easier to debug
        #     ifmap = np.reshape(np.arange(image_size[0]*image_size[1]*in_chn), (image_size[0],image_size[1],in_chn)).astype(np.int64)
        #     weights = np.reshape(np.arange(filter_size[0]*filter_size[1]*in_chn*out_chn),
        #                          (filter_size[0],filter_size[1],in_chn,out_chn)).astype(np.int64)
        #     bias = np.arange(out_chn).astype(np.int64)
        # else:
        #     ifmap = np.random.normal(0, 10, (image_size[0], image_size[1],
        #         in_chn)).astype(np.int64)
        #     weights = np.random.normal(0, 10, (filter_size[0], filter_size[1], in_chn,
        #         out_chn)).astype(np.int64)
        #     bias = np.random.normal(0, 10, out_chn).astype(np.int64)
        



        # Randomly zeros out a number of weights to ensure 
        if (do_premature_prune):
            print("WARNING: Doing a premature prune!")
            print("This will give inaccurate outputs")
            print("Use only for validation of architecture")
            print("------------------------------------------------------------------")
            if (self.num_nonzero < self.block_size or not keep_max):
                for x in range(image_size[0]):
                    for y in range(image_size[1]):
                        for c in range(in_chn//self.block_size):
                            num_to_keep = self.num_nonzero if keep_max else np.random.randint(0,self.num_nonzero+1)
                            to_keep = np.random.choice(self.block_size, num_to_keep, replace=False)
                            cmin = c*self.block_size
                            cmax = cmin + self.block_size
                            ifmap[x][y][cmin:cmax] = [ifmap[x][y][c] if c in to_keep else 0 for c in range(cmin, cmax)]
        else:
            pass
            #print("Don't worry about validation failure messages if using pruning")
        #print("From stimulus:")
        #print("ifmap")
        #print(ifmap)
        #print("weights")
        #print(weights)
        #print("bias")
        #print(bias)
        self.ofmap = np.zeros((image_size[0], image_size[1],
            out_chn))#.astype(np.int64)

        # Reference Output
        reference = conv(ifmap, weights, bias)

        self.serializer.configure(ifmap, weights, bias, in_chn, out_chn, image_size, filter_size)
        self.deserializer.configure(self.ofmap, reference, image_size, out_chn)
