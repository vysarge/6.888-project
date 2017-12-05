from nnsim.module import Module, HWError
from nnsim.reg import Reg
from nnsim.channel import Channel


class PE(Module):
    def instantiate(self, loc_x, loc_y,
            ifmap_chn, filter_chn,
            psum_in_chn, psum_out_chn):
        # PE identifier (immutable)
        self.loc_x = loc_x
        self.loc_y = loc_y
        
        self.stat_type = 'aggregate'
        self.raw_stats = {'pe_mac' : 0}

        # IO channels
        self.ifmap_chn = ifmap_chn
        self.filter_chn = filter_chn
        self.psum_in_chn = psum_in_chn
        self.psum_out_chn = psum_out_chn

        # PE controller state (set by configure)
        self.fmap_per_iteration = 0
        self.num_iteration = 0

        self.fmap_idx = None
        self.iteration = None
        
        # Added an accumulator value here rather than routing everything back through the buffer: no reason to in hardware
        self.value = Reg(0)

    def configure(self, fmap_per_iteration, num_iteration):
        self.fmap_per_iteration = fmap_per_iteration
        self.num_iteration = num_iteration

        self.fmap_idx = 0
        self.iteration = 0

    # Added method to reset internal accumulator
    def reset(self):
        self.value.reset()
    
    def tick(self):
        if (self.iteration == self.num_iteration):
            return
        #print("{},{}".format(self.loc_y, self.loc_x))
        if not (self.fmap_idx == 0):
            #print("{},{}: {} ifmap and {} filter".format(self.loc_y, self.loc_x, self.ifmap_chn.valid(), self.filter_chn.valid()))
            if (self.ifmap_chn.valid() and self.filter_chn.valid()):
                if (not self.value.rd()[0] == self.ifmap_chn.peek()[0]):
                    #print("{},{} sending output {}".format(self.loc_y, self.loc_x, self.value.rd()))
                    self.psum_out_chn.push([self.value.rd()[0], self.value.rd()[1]])
                    #print("{},{} push".format(self.loc_y, self.loc_x))
                    self.fmap_idx = 0
                    self.iteration += 1
                else:
                    loc_tag, ifmap = self.ifmap_chn.pop()
                    #print("{} == {}".format(self.value.rd()[0], loc_tag))
                    assert(self.value.rd()[0] == loc_tag)
                    weight = self.filter_chn.pop()
                    
                    self.raw_stats['pe_mac'] += 1

                    #if (self.loc_y == 0 and self.loc_x == 0):
                    #    print("({},{}, loc {}): {} + {} * {} = {}".format(self.loc_y, self.loc_x,\
                    #        loc_tag, self.value.rd()[1], weight, ifmap,\
                    #        self.value.rd()[1]+ifmap*weight))
                    
                    self.value.wr([loc_tag, self.value.rd()[1]+ifmap*weight])
                    
                
                   

        if (self.fmap_idx == 0):
            if (self.psum_in_chn.valid() and self.ifmap_chn.valid() and self.filter_chn.valid()):
                in_psum = self.psum_in_chn.pop()
                loc_tag, ifmap = self.ifmap_chn.pop()
                weight = self.filter_chn.pop()
                
                self.raw_stats['pe_mac'] += 1
                self.fmap_idx += 1

                self.value.wr([loc_tag, in_psum+ifmap*weight])
                