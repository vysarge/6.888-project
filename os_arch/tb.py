from nnsim.module import Module
from nnsim.channel import Channel
from osarch import OSArch
from stimulus import Stimulus
import numpy as np

#debug = True # allows switching between a main testing conv and a debug one
debugStimulus = False # if true, uses integer and increasing values for inputs, weights, and biases
keepMaxValues = True # if true, always keeps num_nonzero values in each block
do_premature_prune = True # if true, will pre-prune inputs to allow output deserializer to do correct validation
# For actual test runs, should be False, False, True, False

class OSArchTB(Module):
    def instantiate(self, image_size, filter_size, in_chn, out_chn, block_size, ifmap, weights, bias, pruner_name, num_nonzero):
        self.name = 'tb'
        
        # if (debug):
        #     self.image_size = (4, 4)
        #     self.filter_size = (3, 3)
        #     self.in_chn = 2
        #     self.out_chn = 4
        #     self.block_size = 2
        #     self.num_nonzero = 1  #number of non-zero values in each blok, help test the correctness of the arch
        # else:
        #     self.image_size = (16, 16)
        #     self.filter_size = (3, 3)
        #     self.in_chn = 16
        #     self.out_chn = 8
        #     self.block_size = 4
        #     self.num_nonzero = 4
        
        self.image_size = image_size
        self.filter_size = filter_size
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.block_size = block_size
        self.num_nonzero = num_nonzero  #number of non-zero values in each blok, help test the correctness of the arch

        #the inputs to this specific layer
        self.ifmap = ifmap
        self.weights = weights
        self.bias = bias
        self.pruner_name = pruner_name

        self.arr_y = self.out_chn
        self.input_chn = Channel()
        self.output_chn = Channel()

        ifmap_glb_depth = (self.filter_size[1] + (self.filter_size[0]-1)*\
            self.image_size[1]) * self.in_chn // self.block_size
        psum_glb_depth = self.out_chn // self.block_size
        weight_glb_depth = self.filter_size[0]*self.filter_size[1]* \
                self.in_chn*self.out_chn//self.block_size

        self.stimulus = Stimulus(self.arr_y, self.block_size, self.num_nonzero,
            self.input_chn, self.output_chn, self.pruner_name )
        self.dut = OSArch(self.arr_y, self.input_chn, self.output_chn, self.block_size,
            self.num_nonzero, ifmap_glb_depth, psum_glb_depth, weight_glb_depth)

        self.configuration_done = False


    def tick(self):
        if not self.configuration_done:
            self.stimulus.configure(self.image_size, self.filter_size, self.in_chn, self.out_chn, self.ifmap, self.weights, self.bias, debugStimulus, keepMaxValues, do_premature_prune)
            self.dut.configure(self.image_size, self.filter_size, self.in_chn, self.out_chn)
            self.configuration_done = True





#simulate for a single conv layer with all needed inputs
def one_layer(image_size, filter_size, in_chn, out_chn, block_size, ifmap, weights, bias, num_nonzero, pruner_name = "NaivePruner"):
    os_tb = OSArchTB(image_size, filter_size, in_chn, out_chn, block_size, ifmap, weights, bias, pruner_name, num_nonzero)
    run_tb(os_tb, verbose=False)
    return os_tb.stimulus.ofmap



if __name__ == "__main__":
    from nnsim.simulator import run_tb

    ######## fake inputs #########################
    num_layer = 3
    filter_size = [(3,3) for layer in range(num_layer)]
    image_size = [(4,4) for layer in range(num_layer)]
    in_chn = [2 for layer in range(num_layer)]
    out_chn = [4 for layer in range(num_layer)]
    block_size = [2 for layer in range(num_layer)]
    num_nonzero =[1 for layer in range(num_layer)]

    ifmap_l = [np.random.normal(0, 10, (image_size[0][0], image_size[0][1],
        in_chn[0])).astype(np.int64)for layer in range(num_layer)]

    weights = [np.random.normal(0, 10, (filter_size[0][0], filter_size[0][1], in_chn[0],
        out_chn[0])).astype(np.int64)for layer in range(num_layer)]

    bias = [np.random.normal(0, 10, out_chn[0]).astype(np.int64) for layer in range(num_layer)]

    #pruner_name = "NaivePruner"
    pruner_name = "ThresholdPruner"
    pruner_name = "ClusteredPruner"
    ###############################################

    #consecutive_conv_layers:
    #       True: no pooling or other layers between the conv layers, the simulator takes in the very fisrt ifmap and run consecutively
    #       False: each layer's ifmap is supplied in the ifmap list
    
    #consecutive_conv_layers = False
    consecutive_conv_layers = True

    if consecutive_conv_layers:
        print("assuming the simulator is running consective conv layers")
    else:
        print("inconsecutive conv layers: each layer's ifmap is supplied by the user")
    
    for l in range(num_layer):
        if l == 0:
            ifmap = ifmap_l[l]
        ofmap = one_layer(image_size[l], filter_size[l], in_chn[l], out_chn[l], block_size[l], ifmap, weights[l], bias[l],  num_nonzero[l], pruner_name)
        if consecutive_conv_layers:
            ifmap = ofmap
        else:
            ifmap = ifmap_l[l+1]




