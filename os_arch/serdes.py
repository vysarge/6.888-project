from nnsim.module import Module
from nnsim.reg import Reg
from nnsim.simulator import Finish
from nnsim.channel import Channel

import sys
sys.path.append('../pruning')

from naive_pruner import NaivePruner
from converter import Converter

import numpy as np

class InputSerializer(Module):
    def instantiate(self, arch_input_chn, arr_y, block_size, num_nonzero):
        # PE static configuration (immutable)
        #self.arr_x = arr_x
        self.arr_y = arr_y
        #self.chn_per_word = chn_per_word
        self.block_size = block_size
        self.num_nonzero = num_nonzero
        
        self.convert_chn = Channel()
        self.prune_chn = Channel()
        self.arch_input_chn = arch_input_chn

        # Although both InputSerializer and pruner will be pushing to arch_input_chn
        # There is no conflict issue because all weights will be pushed by IS first
        # then all inputs by pruner
        self.converter = Converter(self.convert_chn, self.prune_chn, \
            self.block_size, self.block_size)
        self.pruner = NaivePruner(self.prune_chn,self.arch_input_chn, \
            self.num_nonzero,True)

        self.ifmap = None
        self.weights = None
        self.bias = None

        self.image_size = (0, 0)
        self.filter_size = (0, 0)

        self.ifmap_psum_done = True
        self.pass_done = Reg(False)

        # State Counters
        self.curr_set = 0
        self.curr_filter = 0
        self.iteration = 0
        self.fmap_idx = 0
        self.curr_chn = 0
        self.curr_x = 0 # run through first two dimensions of input
        self.curr_y = 0
        self.bias_set = 0
        #self.send_bias = False

    def configure(self, ifmap, weights, bias, in_chn, out_chn, image_size, filter_size):
        self.ifmap = ifmap
        self.weights = weights
        self.bias = bias

        self.in_chn = in_chn
        self.out_chn = out_chn

        self.image_size = image_size
        self.filter_size = filter_size

        
        
        self.ifmap_psum_done = False
        self.weights_done = False
        self.pass_done.wr(False)
        
        # State Counters
        self.curr_set = 0
        self.curr_filter = 0
        self.iteration = 0
        self.curr_chn = 0
        self.curr_x = 0 # run through first two dimensions of input
        self.curr_y = 0
        self.bias_set = 0
        #self.send_bias = False

    def tick(self):
        if self.pass_done.rd():
            return

        if self.ifmap_psum_done:
            if self.convert_chn.vacancy():
                data = np.zeros(self.block_size)
                self.convert_chn.push(data)
            return

        in_sets = self.in_chn // self.block_size
        out_sets = self.out_chn // self.block_size 
        num_iteration = self.filter_size[0]*self.filter_size[1]

        # read and hold all weights at the beginning for ease of implementation
        if not self.weights_done:
            f_x = self.iteration // self.filter_size[0]
            f_y = self.iteration % self.filter_size[0]

            # Push filters to PE columns. (PE is responsible for pop)
            if self.arch_input_chn.vacancy() and self.iteration < num_iteration:
                cmin = self.curr_filter*self.block_size
                cmax = cmin + self.block_size
                data = np.array([self.weights[f_x, f_y, self.curr_chn, c] \
                        for c in range(cmin, cmax) ])
                #print("{},{},{},{}-{}".format(f_x,f_y,self.curr_chn,cmin,cmax))
                #print(data)
                self.arch_input_chn.push(data) # Gives groups of four along num_filters axis

                
                self.curr_filter += 1
                if (self.curr_filter == out_sets): # Loop through blocks of filters
                    self.curr_filter = 0
                    self.curr_chn += 1
                if (self.curr_chn == self.in_chn): # Loop through channels
                    self.curr_chn = 0
                    self.iteration += 1
                if (self.iteration == num_iteration): # Loop through 2D filter support
                    self.iteration = 0
                    #print("Weights done")
                    self.weights_done = True
                
                #self.curr_set += 1
                #if self.curr_chn == in_sets: # Loops through sets, then filter elements, then filters
                #    self.curr_chn = 0
                #    self.iteration += 1
                #if self.iteration == num_iteration:
                #    self.curr_filter += 1
                #    self.iteration = 0
                #if self.curr_filter == self.arr_x:
                #    self.curr_filter = 0
                #    self.curr_set = 0
                #    self.weights_done = True
        elif self.arch_input_chn.vacancy() and self.bias_set < out_sets:
            cmin = self.bias_set*self.block_size
            cmax = cmin + self.block_size
            data = np.array([ self.bias[c] for c in range(cmin, cmax) ])
            #print("bias")
            #print(data)
            self.arch_input_chn.push(data)
            self.bias_set += 1
        elif not self.ifmap_psum_done:
            if self.convert_chn.vacancy():
                cmin = self.curr_set*self.block_size
                cmax = cmin + self.block_size
                
                #xmin = x
                #xmax = x+self.arr_x
                # Write ifmap to glb
                #data = np.array([ self.ifmap[x, self.curr_y, self.curr_chn] for x in range(xmin, xmax) ])
                data = np.array([ self.ifmap[self.curr_x, self.curr_y, c] for c in range(cmin, cmax) ])
                #print("{},{},{}-{}".format(self.curr_x, self.curr_y, cmin, cmax))
                #print(data)
                
                self.curr_set += 1
                if (self.curr_set == in_sets):
                    self.curr_set = 0
                    self.curr_y += 1
                if (self.curr_y == self.image_size[1]):
                    self.curr_y = 0
                    self.curr_x += 1
                    #if (self.curr_x % self.arr_y == 0):
                    #    self.send_bias = True
                #if (self.curr_x == self.image_size[0]):
                #    self.send_bias = True
                        
                #else:
                #    cmin = self.curr_set*self.chn_per_word
                #    cmax = cmin + self.chn_per_word
                #    # Write bias to glb
                #    data = np.array([ self.bias[c] for c in range(cmin, cmax) ])
                #    #print("bias")
                #    #print(data)
                #    self.curr_set += 1
                #    if (self.curr_set == out_sets):
                #        self.curr_set = 0
                #        self.send_bias = False
                #print(data)
                self.convert_chn.push(data)

                if (self.curr_x == self.image_size[0]):
                    self.curr_x = 0
                    self.ifmap_psum_done = True
                    print("InputSerializer: Inputs and biases written") 
                    print("Continue flushing with zeros")       
        


class InputDeserializer(Module):
    def instantiate(self, arch_input_chn, ifmap_chn, weights_chn, psum_chn,
            arr_y, block_size, num_nonzero):
        
        self.arr_y = arr_y
        self.block_size = block_size
        self.num_nonzero = num_nonzero

        self.stat_type = 'aggregate'
        self.raw_stats = {'dram_rd' : 0}

        self.arch_input_chn = arch_input_chn
        self.ifmap_chn = ifmap_chn
        self.weights_chn = weights_chn
        self.psum_chn = psum_chn

        self.image_size = (0, 0)
        self.filter_size = (0, 0)

        self.fmap_idx = 0
        self.curr_set = 0
        self.num_weights = 0
        self.bias_done = False

    def configure(self, image_size, filter_size, in_chn, out_chn):
        self.image_size = image_size
        self.filter_size = filter_size
        self.in_chn = in_chn
        self.out_chn = out_chn

        self.fmap_idx = 0
        self.curr_set = 0
        self.num_weights = 0 # goes from 0 to the total number of blocks of weights we receive

    def tick(self):
        
        in_sets = self.image_size[1]*self.in_chn // self.block_size # number of sets of inputs to send
        out_sets = self.arr_y // self.block_size # number of sets of biases to send
        fmap_per_iteration = self.image_size[0] # number of times to send sets

        if self.num_weights < self.filter_size[0]*self.filter_size[1]*self.in_chn*self.out_chn//self.block_size:
            target_chn = self.weights_chn
            target_str = 'weights'
        elif self.fmap_idx < fmap_per_iteration:
            if not self.bias_done and self.curr_set < out_sets:#self.curr_set < in_sets:
                target_chn = self.psum_chn
                target_str = 'psum'
            else:
                target_chn = self.ifmap_chn
                target_str = 'ifmap'
                if not self.bias_done:
                    self.bias_done = True
                    self.curr_set = 0
            self.curr_set += 1
        else:
            return

        if self.arch_input_chn.valid():
            if target_chn.vacancy():
                data = [e for e in self.arch_input_chn.pop()]
                target_chn.push(data)
                #print(target_str)
                #print(data)
                self.raw_stats['dram_rd'] += len(data)
                
                if (self.num_weights < self.filter_size[0]*self.filter_size[1]*self.in_chn*self.out_chn//self.block_size):
                    self.num_weights += 1
                elif self.fmap_idx < fmap_per_iteration:
                    if self.curr_set == (in_sets):
                        self.curr_set = 0
                        self.fmap_idx += 1

class OutputSerializer(Module):
    def instantiate(self, arch_output_chn, psum_chn):
        self.arch_output_chn = arch_output_chn
        self.psum_chn = psum_chn
        
        self.stat_type = 'aggregate'
        self.raw_stats = {'dram_wr' : 0}


    def configure(self):
        pass

    def tick(self):
        if self.psum_chn.valid():
            if self.arch_output_chn.vacancy():
                data = [e for e in self.psum_chn.pop()]
                self.arch_output_chn.push(data)
                self.raw_stats['dram_wr'] += len(data)

class OutputDeserializer(Module):
    def instantiate(self, arch_output_chn, arr_y, block_size, num_nonzero):
        # PE static configuration (immutable)
        self.arr_y = arr_y
        self.block_size = block_size
        self.num_nonzero = num_nonzero

        self.arch_output_chn = arch_output_chn

        self.ofmap = None
        self.reference = None

        self.image_size = (0, 0)

        self.curr_set = 0
        self.fmap_idx = 0
        
        self.pass_done = Reg(False)

    def configure(self, ofmap, reference, image_size, out_chn):
        self.ofmap = ofmap
        self.reference = reference
        self.out_chn = out_chn

        self.image_size = image_size

        self.curr_set = 0
        self.fmap_idx = 0

        self.pass_done.wr(False)

    def tick(self):
        if self.pass_done.rd():
            return

        out_sets = self.out_chn//self.block_size
        fmap_per_iteration = self.image_size[0]*self.image_size[1]

        if self.arch_output_chn.valid():
            rcvd = self.arch_output_chn.pop()
            loc_tag = [e[0] for e in rcvd]
            data = [e[1] for e in rcvd]

            #print(loc_tag)
            
            x = loc_tag[0] // self.image_size[1]
            y = loc_tag[0] % self.image_size[1]
            #x = self.fmap_idx % self.image_size[0]
            #y = self.fmap_idx // self.image_size[0]
            self.fmap_idx = x+y*self.image_size[0]
            
            #print("{},{} received (output deserializer)".format(x,y))
            #print(data)

            if self.curr_set < out_sets:
                cmin = self.curr_set*self.block_size
                cmax = cmin + self.block_size
                for c in range(cmin, cmax):
                    assert(self.ofmap[x,y,c] == 0) # should never replace an existing value
                    self.ofmap[x, y, c] = data[c-cmin]
            self.curr_set += 1

            if self.curr_set == out_sets:
                self.curr_set = 0
                self.fmap_idx += 1
            if self.fmap_idx == fmap_per_iteration:
                self.fmap_idx = 0
                self.pass_done.wr(True)
                if np.all(self.ofmap == self.reference):
                    raise Finish("Success")
                else:
                    print("Output")
                    print(self.ofmap)
                    print("Reference")
                    print(self.reference)
                    print("Diff")
                    print(self.ofmap-self.reference)
                    raise Finish("Validation Failed")

