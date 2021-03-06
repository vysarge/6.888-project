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
        self.out_marker = 0


        # Vars for serial output
        self.prev_vals = []
        
        # Buffer.  Use for accumulating input values
        self.buffer = [[0 for j in range(self.input_size)] for i in range(self.in_sets)]
        self.buffer_valid = False

        # channels
        self.in_chn = in_chn
        self.out_chn = out_chn

    def tick(self):
        #print(self.out_marker)
        #print("{} == {}".format(((self.curr_set + 1) % self.in_sets),self.out_marker // self.input_size))
        # Once data is accumulated (placed first to prevent this from happening on the same cycle)
        #if not (((self.out_marker) % self.block_size) // self.input_size == self.curr_set):
        if (self.buffer_valid and self.out_chn.vacancy()):
            # Do this on the last cycle
            if (self.out_marker == self.block_size):
                if (len(self.prev_vals) == 0): # guarantee we will always send /something/ per block
                    self.out_chn.push([0,0,1])
                    #print("Converter output:")
                    #print([0,0,1])
                else:
                    self.prev_vals.append(1)
                    self.out_chn.push(self.prev_vals)
                    #print("Converter output:")
                    #print(self.prev_vals)

                # Go to next block
                self.prev_vals = []
                self.out_marker = 0
                self.buffer_valid = False
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
                        #print("Converter output:")
                        #print(self.prev_vals)
                    self.prev_vals = [self.out_marker, curr_val]
                    self.out_marker += 1
                elif (curr_val == 0):
                    self.out_marker += 1




        # Accumulate data for pruning etc
        #if (((self.curr_set + 1) % self.in_sets) == self.out_marker // self.input_size and not ((self.curr_set + 1) % self.in_sets)==0):
        #    pass
        if (self.in_chn.valid() and not self.buffer_valid):
            data = self.in_chn.pop()
            #print(data)
            self.buffer[self.curr_set] = data
            self.curr_set += 1
            if (self.curr_set == self.in_sets):
                self.buffer_valid = True
                self.curr_set = 0


