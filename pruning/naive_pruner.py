from nnsim.module import Module
from nnsim.reg import Reg
from nnsim.channel import Channel

class NaivePruner(Module):
    def instantiate(self, in_chn, out_chn, num_nonzero, preserve_order=True):
        self.name = 'naive_pruner'

        # num_nonzero is the number of nonzero activations to keep
        # The rest will be pruned blockwise
        self.num_nonzero = num_nonzero

        # boolean var: whether to preserve order in outputs
        # if false, there is no guarantee that activations will be sent in the 'correct' order
        self.preserve_order = preserve_order

        # position tracking vars
        self.valid_marker = 0 # The current number of valid activations (of num_nonzero)
        self.prev_valid_marker = 0 # The number of valid activations in the just-completed buffer
        self.write_to = 0 # 0 or 1; which of the two buffers to write to
        self.read_from = 0 # 0 or 1; which of the two buffers to read from
        # writing always has priority if they are the same
        # but write_to cannot switch to read_from's value at any point
        # to prevent overwriting values which have not yet been read
        self.ready_to_switch = False # used to indicate that write_to should switch when possible
        self.out_marker = 0 # The next activation to output

        # data structure for storing running nonzero activations
        self.buffer = [[0 for i in range(self.num_nonzero)] for j in [0,1]]

        # channels
        self.in_chn = in_chn
        self.out_chn = out_chn

    def tick(self):
        # Send values to output channel from buffer
        if not (self.read_from == self.write_to) and self.out_chn.vacancy():
            data = self.buffer[self.read_from][self.out_marker]
            if (self.out_marker == self.prev_valid_marker-1):
                # if reached the last valid activation, switch to the other buffer
                self.out_marker = 0
                self.read_from = 1-self.read_from
                data.append(1) # make sure to add termination bit
            else:
                data.append(0)
                self.out_marker += 1
            self.out_chn.push(data)
            
        
        # Load values from input channel into buffer
        # and perform pruning
        if (self.in_chn.valid() and not self.ready_to_switch):
            data = self.in_chn.pop()
            if (self.valid_marker < self.num_nonzero): # data[2] no longer matters
                self.buffer[self.write_to][self.valid_marker] = [data[0], data[1]]
                self.valid_marker += 1
            else: # If there are more activations than desired, replace the lowest
                dummy = lambda x: self.buffer[self.write_to][x][1] # give value of activation
                x = range(self.num_nonzero)
                replace_ind = min(x, key=dummy)
                if (self.preserve_order):
                    self.buffer[self.write_to] = [self.buffer[self.write_to][i] \
                        if i < replace_ind else self.buffer[self.write_to][i+1] \
                        for i in range(self.num_nonzero-1)]
                    self.buffer[self.write_to].append([data[0], data[1]])
                else:
                    self.buffer[self.write_to][replace_ind] = [data[0], data[1]]
            if (data[2] == 1):
                self.ready_to_switch = True
            #print(self.buffer[self.write_to])

        # switch which buffer to write to
        # (makes it easier to avoid overwiting unread data)
        if (self.ready_to_switch and self.read_from == self.write_to):
            self.write_to = 1-self.write_to
            self.prev_valid_marker = self.valid_marker
            self.valid_marker = 0
            self.ready_to_switch = False



