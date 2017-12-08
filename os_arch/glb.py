from nnsim.module import Module
from nnsim.ram import SRAM, RD, WR
from nnsim.channel import Channel

import numpy as np

class IFMapWeightsGLB(Module):
    def instantiate(self, ifmap_wr_chn, ifmap_rd_chn, weights_wr_chn, weights_rd_chn,\
            arr_y, ifmap_glb_depth, weights_glb_depth, \
            block_size, num_nonzero):
        self.ifmap_wr_chn = ifmap_wr_chn
        self.ifmap_rd_chn = ifmap_rd_chn
        self.weights_wr_chn = weights_wr_chn
        self.weights_rd_chn = weights_rd_chn
        self.arr_y = arr_y
        self.block_size = block_size
        self.num_nonzero = num_nonzero
        self.name = 'ifmap_weights_glb'

        self.in_chn = 0
        self.out_chn = 0
        
        self.stat_type = 'show'
        self.raw_stats = {'size' : (ifmap_glb_depth, num_nonzero*3), 'rd': 0, 'wr': 0}


        self.isram = SRAM(ifmap_glb_depth, num_nonzero*3)
        self.ilast_read = Channel(3)
        self.ifmap_glb_depth = ifmap_glb_depth

        self.wsram = SRAM(weights_glb_depth, block_size)
        self.wlast_read = Channel(1)
        # Channel depth of one here prevents SRAM reads from colliding
        # was having issues with a later read 'replacing' an earlier one
        # and thus getting the wrong data
        # having only one extant write on an SRAM at a time prevents this
        self.weights_glb_depth = weights_glb_depth

        # Channel to hold indices of weights that need to be sent
        # to NoC
        self.weights_to_send = Channel(3)

        self.image_size = (0, 0)
        self.filter_size = (0, 0)
        self.fmap_sets = 0
        self.fmap_per_iteration = 0

        self.curr_set = 0
        self.fmap_idx = 0
        self.iteration = 0
        self.iwr_done = False
        self.wwr_done = False
        
        # For managing convolution
        self.curr_x = 0
        self.curr_y = 0
        self.curr_chn = 0
        self.request_idx = 0
        self.send_idx = 0
        #self.curr_filt_x = 0
        #self.curr_filt_y = 0
        self.ifmap_done = False

        # for weights
        self.addr = 0
        self.base_addr = 0 # to store values from self.weights_to_send
        self.base_addr_wo_chn = -1 # to keep track of current position within 3x3 filter

        # invalid weights and inputs to use at the end to flush out last outputs
        self.weights_to_flush = 0
        self.inputs_to_flush = 0

        self.needed_addr = 0
        self.ready_to_output = False # ready to output a filter_size block of inputs
        self.curr_data = [0 for i in range(3*num_nonzero)]
        self.curr_weights = [0 for i in range(block_size)]
        self.data_idx = num_nonzero # block other operations while actively working through data
        # send one data point at a time (of num_nonzero)

    def configure(self, image_size, filter_size, in_chn, out_chn, fmap_per_iteration):
        self.wr_done = False

        self.image_size = image_size
        self.filter_size = filter_size
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.fmap_per_iteration = fmap_per_iteration
        
        # For managing convolution
        self.curr_x = 0
        self.curr_y = 0
        self.curr_chn = 0
        self.request_idx = 0
        self.send_idx = 0
        self.curr_filt_x = 0
        self.curr_filt_y = 0
        self.curr_filt_set = 0
        self.ifmap_done = False

        offset_x = (self.filter_size[0] - 1)//2
        offset_y = (self.filter_size[1] - 1)//2
        # The first address needed to be filled in order to start sending
        #self.needed_addr = (self.image_size[0]*(1+offset_y) + 1+offset_x) *\
        #    (self.in_chn // self.block_size) - 1
        self.needed_addr = (self.image_size[0]*(offset_y) + 1+offset_x) *\
            (self.in_chn // self.block_size) - 1
        # Goes high to transfer sram control to output
        # Doing them synchronously would be better, but complicates things
        self.ready_to_output = False

    def tick(self):

        # WEIGHTS-------------------------------------------------------------------
        num_iterations = self.image_size[0]*self.image_size[1]*self.in_chn//self.block_size
        max_addr = self.filter_size[0]*self.filter_size[1]*self.in_chn*self.out_chn//self.block_size
        
        verbose = False
        
        if not self.wwr_done:
            if self.weights_wr_chn.valid():
                data = self.weights_wr_chn.pop()
                self.raw_stats['wr'] += len(data)
                #print(self.addr)
                #print("weights (iw glb) at addr {}".format(self.addr))
                #print(data)
                #print(self.addr)
                self.wsram.request(WR, self.addr, np.asarray(data))
                self.addr += 1
                #print("storing weights (wi glb)")
                #print(data)
                if (self.addr == max_addr):
                    self.addr = self.out_chn // self.block_size
                    self.wwr_done = True
                    #print("Done storing weights (wi glb)")
                    #print("--------------------------")
                    #print(self.wsram.data)
                    #print("--------------------------")
                

        # within this block of code self.addr is re-used
        # here it is more analogous to curr_set
        # and refers to the current block of filters (last index of the four)
        # that is being read
        else:
            # Catch addresses that correspond to nonzero inputs
            # search "self.weights_to_send.push(waddr)" below
            if (self.weights_to_send.valid() and self.addr == self.out_chn // self.block_size):
                self.base_addr = self.weights_to_send.pop()
                self.addr = 0
            # cycle through channels using self.addr
            # make requests to memory; will pick these up in the next if statement below
            elif (self.wlast_read.vacancy() and not self.addr == self.out_chn // self.block_size):
                full_addr = self.base_addr + self.addr
                self.wsram.request(RD, full_addr)
                self.wlast_read.push(False)
                #print("Request weights (wi glb):")
                #print(full_addr)
                self.addr += 1
        # catch requests from memory; send results to WeightsNoC
        if self.wlast_read.valid() and self.weights_rd_chn.vacancy(1):
            is_zero = self.wlast_read.pop()
            data = [e for e in self.wsram.response()]
            self.weights_rd_chn.push(data)
            #print("weights sent (from iw glb)")
            #print(data)
            self.raw_stats['rd'] += len(data)

        # these two if statements take care of an issue that occurs at the end
        # PEs don't automatically detect the end of the computation without inputs
        # from another location
        # So we send in some dummy inputs to flush out the last outputs
        if self.weights_rd_chn.vacancy(1) and not self.wlast_read.valid() and \
            not self.weights_to_send.valid() and self.addr == self.out_chn // self.block_size\
            and self.weights_to_flush > 0:
            self.weights_to_flush -= 1
            self.weights_rd_chn.push([0 for i in range(self.block_size)])
        if self.ifmap_done and self.inputs_to_flush > 0 and self.ifmap_rd_chn.vacancy(1):
            self.inputs_to_flush -= 1
            self.ifmap_rd_chn.push([-1, 0, 0])

        # IFMAP-------------------------------------------------------------------
        if not (self.ifmap_done and not self.ilast_read.valid() and not self.ready_to_output):
            verbose = False
            
            # shorthand values that will be useful later
            num_iteration = self.filter_size[0]*self.filter_size[1]
            offset_x = (self.filter_size[0] - 1)//2
            offset_y = (self.filter_size[1] - 1)//2
            filter_x = self.iteration % self.filter_size[0] - offset_x
            filter_y = self.iteration // self.filter_size[0] - offset_y
            in_sets = self.in_chn // self.block_size
            out_sets = self.out_chn // self.block_size
            if not self.iwr_done and not self.ready_to_output:
                # Write to GLB
                if self.ifmap_wr_chn.valid():
                    data = self.ifmap_wr_chn.pop()
                    data = np.reshape(np.asarray(data), (-1))
                    
                    full_addr = in_sets*self.fmap_idx + self.curr_set
                    #print("ifmap (wi glb) received data:")
                    #print(data)
                    #print("{} >?= {}".format(full_addr, self.needed_addr))
                    self.curr_set += 1
                    addr = full_addr % self.ifmap_glb_depth

                    # if we have enough inputs in memory to start sending 
                    if (full_addr == self.needed_addr):
                        self.ready_to_output = True
                        self.needed_addr += in_sets

                    self.isram.request(WR, addr, data)
                    self.raw_stats['wr'] += len(data)
                    #print("ifmap, iw glb")
                    #print("{} written to {}".format(data, addr))

                    if self.curr_set == self.fmap_sets:
                        self.curr_set = 0
                        self.fmap_idx += 1
                    if self.fmap_idx == self.fmap_per_iteration:
                        # Done initializing ifmaps and psums
                        print("iw glb: Finished filling ifmap buffer")
                        self.fmap_idx = 0
                        self.iwr_done = True
            elif self.ready_to_output:
                increment_vals = False
                # send data to NoC
                if (self.ilast_read.valid() and self.ifmap_rd_chn.vacancy(1)):
                    is_zero = self.ilast_read.pop()
                    #print(is_zero)
                    if (not is_zero):
                        self.curr_data = [e for e in self.isram.response()]
                        self.data_idx = 0
                    else:
                        increment_vals = True
                elif (not self.data_idx == self.num_nonzero and self.weights_to_send.vacancy() and self.base_addr_wo_chn >= 0):
                    data = [self.curr_data[i] for i in \
                        range(self.data_idx*3, self.data_idx*3 + 3)]
                    data_mod = [self.curr_x*self.image_size[1]+self.curr_y,\
                        data[1], data[2]]
                    self.ifmap_rd_chn.push(data_mod)
                    #print("iw glb inputs sent: {},{},{}".format(self.curr_x, self.curr_y, self.curr_chn))
                    #print(data_mod)
                    self.raw_stats['rd'] += 1

                    # Assertion checks that we will not attempt to read data that
                    # has not yet been stored in memory
                    waddr = self.base_addr_wo_chn + data[0]*out_sets
                    assert (self.wwr_done or waddr < self.addr)
                    #self.wsram.request(RD, waddr)
                    #self.wlast_read.push(False)
                    self.weights_to_send.push(waddr)
                    #print("Send request (wi glb):")
                    #print(waddr)

                    self.data_idx += 1
                    #if (self.data_idx == self.num_nonzero):
                    if (data[2] == 1):
                        self.data_idx = self.num_nonzero
                        increment_vals = True
                    if (self.data_idx == self.num_nonzero):
                        self.base_addr_wo_chn = -1
                    #print(self.data_idx)
                if (increment_vals):
                    #print(self.send_idx)
                    self.curr_chn += 1
                    if (self.curr_chn == in_sets):
                        self.curr_chn = 0
                        self.send_idx += 1
                    if (self.send_idx == self.filter_size[0]*self.filter_size[1]):
                        #print("Ready to shift input glb frame ({},{})".format(self.curr_x, self.curr_y))
                        self.send_idx = 0
                        self.curr_y += 1
                        if (self.curr_y == self.image_size[1]):
                            self.curr_y = 0
                            self.curr_x += 1
                        if (self.curr_x == self.image_size[0]):
                            self.curr_x = 0
                            self.ifmap_done = True
                            print("Done sending inputs from iw glb")
                            self.ready_to_output = False
                            self.inputs_to_flush = 1
                            self.weights_to_flush = self.arr_y // self.block_size
                        elif (not self.iwr_done):
                            self.ready_to_output = False
                        #print(self.ifmap_wr_chn.valid())
                    
                
                    
                # stage one of these at a time
                # request data from SRAM
                if (not self.ifmap_done and self.ilast_read.vacancy(1) and \
                    self.data_idx == self.num_nonzero and self.weights_to_send.vacancy()\
                    and self.base_addr_wo_chn == -1):
                    # and not (self.curr_x == self.image_size[0]):
                    x_adj = (self.curr_x + self.curr_filt_x - offset_x)
                    y_adj = self.curr_y + self.curr_filt_y - offset_y
                    idx = x_adj*self.image_size[1] + y_adj
                    #print("{},{},{} input requested".format(x_adj, y_adj, self.curr_filt_set))
                    #print(idx)
                    if (x_adj < 0 or x_adj >= self.image_size[0] or y_adj < 0 or y_adj >= self.image_size[1]):
                        self.ilast_read.push(True)
                    else:
                        addr = (idx * in_sets + self.curr_filt_set) % self.ifmap_glb_depth
                        self.isram.request(RD, addr)
                        self.ilast_read.push(False)
                        # set up for corresponding weights to be sent later
                        self.base_addr_wo_chn = self.curr_filt_x*self.filter_size[1]\
                            *self.in_chn*out_sets + \
                            self.curr_filt_y*self.in_chn*out_sets + \
                            self.curr_filt_set*self.block_size*out_sets
                        #print("Next base addr = {}".format(self.base_addr_wo_chn))
                        
                    
                    
                    self.curr_filt_set += 1
                    if (self.curr_filt_set == in_sets):
                        self.curr_filt_set = 0
                        self.curr_filt_y += 1
                    if (self.curr_filt_y == self.filter_size[1]):
                        self.curr_filt_y = 0
                        self.curr_filt_x += 1
                    if (self.curr_filt_x == self.filter_size[0]):
                        self.curr_filt_x = 0


class IFMapGLB(Module):
    def instantiate(self, wr_chn, rd_chn, arr_y, glb_depth, block_size, num_nonzero):
        self.wr_chn = wr_chn
        self.rd_chn = rd_chn
        self.arr_y = arr_y
        self.block_size = block_size
        self.num_nonzero = num_nonzero
        self.name = 'ifmap_glb'
        
        self.stat_type = 'show'
        self.raw_stats = {'size' : (glb_depth, num_nonzero), 'rd': 0, 'wr': 0}


        self.sram = SRAM(glb_depth, num_nonzero*3)
        self.last_read = Channel(3)
        self.glb_depth = glb_depth

        self.image_size = (0, 0)
        self.filter_size = (0, 0)
        self.fmap_sets = 0
        self.fmap_per_iteration = 0

        self.curr_set = 0
        self.fmap_idx = 0
        self.iteration = 0
        self.wr_done = False
        
        # For managing convolution
        self.curr_x = 0
        self.curr_y = 0
        self.curr_chn = 0
        self.request_idx = 0
        self.send_idx = 0
        #self.curr_filt_x = 0
        #self.curr_filt_y = 0
        self.ifmap_done = False

        self.needed_addr = 0
        self.ready_to_output = False # ready to output a filter_size block of inputs
        self.curr_data = [0 for i in range(3*num_nonzero)]
        self.data_idx = num_nonzero # block other operations while actively working through data
        # send one data point at a time (of num_nonzero)

    def configure(self, image_size, filter_size, in_chn, fmap_per_iteration):
        self.wr_done = False

        self.image_size = image_size
        self.filter_size = filter_size
        self.in_chn = in_chn
        self.fmap_per_iteration = fmap_per_iteration
        
        # For managing convolution
        self.curr_x = 0
        self.curr_y = 0
        self.curr_chn = 0
        self.request_idx = 0
        self.send_idx = 0
        self.curr_filt_x = 0
        self.curr_filt_y = 0
        self.curr_filt_set = 0
        self.ifmap_done = False

        offset_x = (self.filter_size[0] - 1)//2
        offset_y = (self.filter_size[1] - 1)//2
        # The first address needed to be filled in order to start sending
        self.needed_addr = (self.image_size[0]*(1+offset_y) + 1+offset_x) *\
            (self.in_chn // self.block_size) - 1
        # Goes high to transfer sram control to output
        # Doing them synchronously would be better, but complicates things
        self.ready_to_output = False

    def tick(self):
        if (self.ifmap_done and not self.last_read.valid() and not self.ready_to_output):
            return
        
        verbose = False
        
        num_iteration = self.filter_size[0]*self.filter_size[1]
        offset_x = (self.filter_size[0] - 1)//2
        offset_y = (self.filter_size[1] - 1)//2
        filter_x = self.iteration % self.filter_size[0] - offset_x
        filter_y = self.iteration // self.filter_size[0] - offset_y
        in_sets = self.in_chn // self.block_size
        #print(filter_x)

        if not self.wr_done and not self.ready_to_output:
            # Write to GLB
            if self.wr_chn.valid():
                data = self.wr_chn.pop()
                data = np.reshape(np.asarray(data), (-1))
                # Write ifmap to glb
                #print("ifmap_glb")
                #print(data)

                full_addr = in_sets*self.fmap_idx + self.curr_set
                self.curr_set += 1
                addr = full_addr % self.glb_depth

                # if we have enough inputs in memory to start sending 
                if (full_addr == self.needed_addr):
                    self.ready_to_output = True
                    self.needed_addr += in_sets

                self.sram.request(WR, addr, data)
                self.raw_stats['wr'] += len(data)
                if self.curr_set == self.fmap_sets:
                    self.curr_set = 0
                    self.fmap_idx += 1
                if self.fmap_idx == self.fmap_per_iteration:
                    # Done initializing ifmaps and psums
                    # self.sram.dump()
                    self.fmap_idx = 0
                    self.wr_done = True
        elif self.ready_to_output:
            # send data to NoC
            if (self.last_read.valid() and self.rd_chn.vacancy(1) and self.data_idx == 0):
                xmin = self.curr_filt_x
                xmax = xmin + self.arr_y
                #print("{}-{},{},{}".format(xmin, xmax, self.holder_y, self.curr_chn))
                #data = [self.holder[x][self.holder_y][self.curr_chn] for x in range(xmin, xmax)]
                is_zero = self.last_read.pop()
                if (not is_zero):
                    self.curr_data = [e for e in self.sram.response()]
                    self.data_idx = 0

            if (not self.data_idx == self.num_nonzero):
                data = [self.curr_data[i] for i in \
                    range(self.data_idx*3, self.data_idx*3 + 3)]
                self.rd_chn.push(data)
                self.raw_stats['rd'] += len(data)
                self.data_idx += 1
                if (self.data_idx == num_nonzero):
                    self.data_idx = 0
                    self.curr_chn += 1
                    if (self.curr_chn == self.arr_y):
                        self.curr_chn = 0
                        self.send_idx += 1
                    if (self.send_idx == self.filter_size[0]*self.filter_size[1]):
                        if (verbose):
                            print("Ready to shift input glb frame")
                        self.send_idx = 0
                        self.curr_y += 1
                        if (self.curr_y == self.image_size[1]):
                            self.curr_y = 0
                            self.curr_x += 1
                        if (self.curr_x == self.image_size[0]):
                            self.curr_x = 0
                            self.ifmap_done = True
                            self.ready_to_output = False
                        elif (not self.wr_done):
                            self.ready_to_output = False

                #print("{},{},{}".format(self.holder_x, self.holder_y, self.curr_chn))
                #print(data)
                
                
                
            # stage one of these at a time
            # request data from SRAM
            if (not self.ifmap_done and self.last_read.vacancy(1) and self.data_idx == num_nonzero):
                # and not (self.curr_x == self.image_size[0]):
                idx = (self.curr_x + self.curr_filt_x - offset_x)*self.image_size[1] + self.curr_y + self.curr_filt_y - offset_y
                #print(idx)
                if (idx >= self.image_size[0]*self.image_size[1] or idx < 0):
                    self.last_read.push(True)
                else:
                    addr = idx * in_sets + self.curr_filt_set
                    self.sram.request(RD, addr)
                    self.last_read.push(False)
                
                self.curr_filt_set += 1
                if (self.curr_filt_set == in_sets):
                    self.curr_filt_set = 0
                    self.curr_filt_y += 1
                if (self.curr_filt_y == self.filter_size[1]):
                    self.curr_filt_y = 0
                    self.curr_filt_x += 1
                if (self.curr_filt_x == self.filter_size[0]):
                    self.curr_filt_x = 0
                
            

class PSumGLB(Module):
    def instantiate(self, dram_wr_chn, noc_wr_chn, rd_chn, glb_depth, block_size, num_nonzero):
        self.dram_wr_chn = dram_wr_chn
        self.noc_wr_chn = noc_wr_chn
        self.rd_chn = rd_chn
        self.name = 'psum_glb'
        self.block_size = block_size
        self.num_nonzero = num_nonzero
        
        self.stat_type = 'show'
        self.raw_stats = {'size' : (glb_depth, block_size), 'rd': 0, 'wr': 0}

        self.sram = SRAM(glb_depth, block_size, nports=2)
        self.last_read = Channel(3)

        self.filter_size = (0, 0)
        self.fmap_sets = 0
        self.fmap_per_iteration = 0

        self.rd_set = 0
        self.fmap_rd_idx = 0
        self.iteration = 0

        self.wr_set = 0
        self.fmap_wr_idx = 0
        self.wr_done = False

    def configure(self, filter_size, out_chn, fmap_per_iteration):
        self.wr_done = False

        self.filter_size = filter_size
        self.out_chn = out_chn
        self.fmap_per_iteration = fmap_per_iteration

        self.rd_set = 0
        self.fmap_rd_idx = 0
        self.iteration = 0

        self.wr_set = 0
        self.fmap_wr_idx = 0
        self.wr_done = False

    def tick(self):
        num_iteration = 1#self.filter_size[0]*self.filter_size[1]

        if not self.wr_done:
            # Write to GLB
            if self.dram_wr_chn.valid():
                data = self.dram_wr_chn.pop()
                # Write ifmap to glb
                #print("psum_glb")
                #print(data)
                #addr = self.fmap_sets*self.fmap_wr_idx + self.wr_set
                addr = self.wr_set
                self.wr_set += 1
                self.sram.request(WR, addr, data, port=0)
                self.raw_stats['wr'] += len(data)
                if self.wr_set == self.out_chn//self.block_size:
                    self.wr_set = 0
                    self.wr_done = True
                    #self.fmap_wr_idx += 1
                #if self.fmap_wr_idx == self.fmap_per_iteration:
                #    # Done initializing ifmaps and psums
                #    # self.sram.dump()
                #    #print("done!")
                #    self.fmap_wr_idx = 0
                #    self.wr_done = True
                
        else:
            # Read from GLB and deal with SRAM latency
            if self.rd_chn.vacancy(1) and self.iteration < num_iteration:
                #addr = self.fmap_sets*self.fmap_rd_idx + self.rd_set
                addr = self.rd_set
                self.sram.request(RD, addr, port=0)
                self.last_read.push(False)
                self.rd_set += 1
                if self.rd_set == self.out_chn//self.block_size:
                    self.rd_set = 0
                    self.fmap_rd_idx += 1
                if self.fmap_rd_idx == self.fmap_per_iteration:
                    self.fmap_rd_idx = 0
                    self.iteration += 1

            # Process the last read sent to the GLB SRAM
            if self.last_read.valid():
                is_zero = self.last_read.pop()
                data = [0]*self.block_size if is_zero else \
                        [e for e in self.sram.response()]
                self.rd_chn.push(data)
                self.raw_stats['rd'] += len(data)

class WeightsGLB(Module):
    def instantiate(self, wr_chn, rd_chn, glb_depth, block_size):
        self.wr_chn = wr_chn
        self.rd_chn = rd_chn
        self.name = 'weight_glb'
        
        self.filter_size = (0,0)
        self.image_size = (0,0)
        self.wr_done = False
        self.iteration = 0
        self.addr = 0
        self.in_chn = 0
        self.out_chn = 0
        #self.arr_y = 0
        #self.out_sets = 0
        self.block_size = block_size
        
        self.sram = SRAM(glb_depth, block_size)
        self.last_read = Channel(3)
        
        self.stat_type = 'show'
        self.raw_stats = {'size' : (glb_depth, block_size), 'rd': 0, 'wr': 0}
        
    def configure(self, filter_size, image_size, in_chn, out_chn):
        self.filter_size = filter_size
        self.image_size = image_size
        self.iteration = 0
        self.addr = 0
        self.in_chn = in_chn
        self.out_chn = out_chn
        #self.arr_y = arr_y
        #self.out_sets = out_sets
        
        self.wr_done = False
        
    def tick(self):
        # num_iterations = times to read out all weights
        # max_addr = number of slots to hold all blocks of weights
        num_iterations = self.image_size[0]*self.image_size[1]*self.in_chn//self.block_size
        max_addr = self.filter_size[0]*self.filter_size[1]*self.in_chn*self.out_chn//self.block_size
        
        verbose = False
        
        if not self.wr_done:
            if self.wr_chn.valid():
                data = self.wr_chn.pop()
                self.raw_stats['wr'] += len(data)
                #print(self.addr)
                #print(data)
                #print(type(data))
                self.sram.request(WR, self.addr, np.asarray(data))
                self.addr += 1
                if (self.addr == max_addr):
                    self.addr = 0
                    self.wr_done = True
                if (verbose):
                    print("weight_glb")
                    print(data)

        
        elif self.rd_chn.vacancy(1):
            if (self.iteration < num_iterations):
                self.sram.request(RD, self.addr)
                self.last_read.push(False)
                self.addr += 1
                if (self.addr == max_addr):
                    self.addr = 0
                    self.iteration += 1
            #self.rd_chn.push(data)
            #self.raw_stats['rd'] += len(data)
        if self.last_read.valid():
            is_zero = self.last_read.pop()
            data = [e for e in self.sram.response()]
            self.rd_chn.push(data)
            #print(self.iteration)
            #print(data)
            self.raw_stats['rd'] += len(data)





