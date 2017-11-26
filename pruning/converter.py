from nnsim.module import Module
from nnsim.reg import Reg
from nnsim.channel import Channel

class Converter(Module):
    def instantiate(self, in_chn, out_chn, input_size, block_size):
        self.name = 'converter'

        # input_size is the width of the input channel
        # block_size is the size of blocks in memory for which we make guarantees
        self.input_size = input_size
        self.block_size = block_size

        # Number of times to transfer the input into the buffer
        self.in_sets = block_size // input_size
        assert(block_size % input_size == 0) # For now, ensure that input is evenly divisible
        self.curr_set = 0 # Counter for where we are currently in the buffer

        # vars for marking where we are in the buffer etc
        #self.in_marker = 0
        self.out_marker = 0


        # Vars for serial output
        #self.transfer_marker = 0
        self.prev_vals = []
        
        # Buffer.  Use for accumulating input values
        self.buffer = [[0 for j in range(self.input_size)] for i in range(self.in_sets)]

        # channels
        self.in_chn = in_chn
        self.out_chn = out_chn

    def tick(self):
        # Once data is accumulated (placed first to prevent this from happening on the same cycle)
        if not (((self.out_marker) % self.block_size) // self.input_size == self.curr_set):
            # Do this on the last cycle
            if (self.out_marker == self.block_size):
                if (len(self.prev_vals) == 0): # guarantee we will always send /something/ per block
                    self.out_chn.push([0,0,1])
                else:
                    self.prev_vals.append(1)
                    self.out_chn.push(self.prev_vals)

                # Go to next block
                self.prev_vals = []
                self.out_marker = 0
            else:
                # pull out and check current value
                # if nonzero and there's room to push to the buffer, do so
                # hold onto values before sending so that we can send the last bit correctly
                # (1 if last nonzero activation in this block, 0 otherwise)
                j = self.out_marker % self.input_size
                i = self.out_marker // self.input_size
                curr_val = self.buffer[i][j]
                if (not curr_val == 0 and self.out_chn.vacancy()):
                    if (len(self.prev_vals) > 0):
                        self.prev_vals.append(0)
                        self.out_chn.push(self.prev_vals)
                    self.prev_vals = [self.out_marker, curr_val]
                    self.out_marker += 1
                elif (curr_val == 0):
                    self.out_marker += 1




        # Accumulate data for pruning etc
        if (self.in_chn.valid() and not \
                ((self.curr_set + 1) % self.in_sets) == self.out_marker // self.input_size):
            data = self.in_chn.pop()
            #imin = self.curr_set * self.input_size
            #imax = imin + self.input_size
            self.buffer[self.curr_set] = data
            self.curr_set += 1
            if (self.curr_set == self.in_sets):
                self.curr_set = 0


