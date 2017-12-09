from nnsim.module import Module
from nnsim.reg import Reg
from nnsim.channel import Channel

class NaivePruner(Module):
    def instantiate(self, in_chn, out_chn, num_nonzero, block_size, preserve_order=True):
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
        #data_out consists of a group of tuples now
        self.data_out = [[0 for i in range(3)] for j in range(self.num_nonzero)]  
        #self.data_out = [0 for i in range(3*self.num_nonzero)]

        # channels
        self.in_chn = in_chn
        self.out_chn = out_chn

    def tick(self):
        # Send values to output channel from buffer
        if not (self.read_from == self.write_to) and self.out_chn.vacancy():
            data = self.buffer[self.read_from][self.out_marker]
            if (self.out_marker == self.prev_valid_marker-1):
                # if reached the last valid activation
                data.append(1) # make sure to add termination bit
                self.data_out[self.out_marker] = data
                # push only if it reaches the last bit
                self.out_chn.push(self.data_out)
                #print("Pruner out:")
                #print(self.data_out)

                # reset counters, switch to the other buffer
                self.data_out = [[0 for i in range(3)] for j in range(self.num_nonzero)]
                self.out_marker = 0
                self.read_from = 1-self.read_from
            else:
                data.append(0)
                self.data_out[self.out_marker] = data
                self.out_marker += 1            

            #self.out_chn.push(data)        
        
        # Load values from input channel into buffer
        # and perform pruning
        if (self.in_chn.valid() and not self.ready_to_switch):
            data = self.in_chn.pop()
            #print(data)
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


#prune out the zeros first, then look to see which data points are good for efficient run length encoding

class ClusteredPruner(Module):
    def instantiate(self, in_chn, out_chn, num_nonzero, block_size, preserve_order=True):
        self.name = 'clustered_pruner'

        # num_nonzero is the number of nonzero activations to keep
        # The rest will be pruned blockwise
        self.num_nonzero = num_nonzero


        self.block_size = block_size

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
        # storage stores the whole block of activations for selecting which ones to prune
        self.storage = [0 for i in range(self.block_size)] 
        self.buffer = [[0 for i in range(self.num_nonzero)] for j in [0,1]]
        self.data_out = [[0 for i in range(3)] for j in range(self.num_nonzero)]
        #self.data_out = [0 for i in range(3*self.num_nonzero)]

        # channels
        self.in_chn = in_chn
        self.out_chn = out_chn

    def tick(self):
        # Send values to output channel from buffer
        if not (self.read_from == self.write_to) and self.out_chn.vacancy():
            data = self.buffer[self.read_from][self.out_marker]
            # prev marker marks how many nonzeros there are in the last round
            if (self.out_marker == self.prev_valid_marker-1):  
                # if reached the last valid activation
                data.append(1) # make sure to add termination bit
                self.data_out[self.out_marker] = data
                self.out_chn.push(self.data_out)
                #print("Pruner out:")
                #print(self.data_out)

                # reset counters, switch to the other buffer
                self.data_out = [[0 for i in range(3)] for j in range(self.num_nonzero)]
                self.out_marker = 0
                self.read_from = 1-self.read_from #toggle the read/write buffer
            else:
                data.append(0)
                self.data_out[self.out_marker] = data
                self.out_marker += 1            

            #self.out_chn.push(data)        
        
        # Load values from input channel into buffer
        # and perform pruning
        if (self.in_chn.valid() and not self.ready_to_switch):
            data = self.in_chn.pop()                    
            #store the data in the storage buffer    
            # print("valid_marker: ", self.valid_marker, "block_size: ",self.block_size)          
            self.storage[self.valid_marker] = [data[0], data[1]]
            self.valid_marker += 1
            if data[2] == 1:   #if this is the last data of the block, need to prune the block and store it to write-to buffer  

                #print(self.storage)  
                #print("valid_marker: ", self.valid_marker)                                  
                if self.valid_marker <= self.num_nonzero:   #if there are less than num_nonzero non-zero numbers
                    self.buffer[self.write_to][0:self.valid_marker-1] = [self.storage[i] for i in range(self.valid_marker)]
               
                else: #if there are more than num_nonzero numbers, prune clustered values

                    num_remove = self.valid_marker - self.num_nonzero #number of values that need to be removed
                    num_left = self.valid_marker                      #number of values that are left in the buffer
                    self.valid_marker = self.num_nonzero

                    #print("num_remove:", num_remove, "num_left: ", num_left)
                    for r in range(num_remove):                       #remove num_remove times in order to remove 
                        max_interval = 0                              #the max interval of zeros 
                        best_idx = 0                                  #best index to remove
                        prev_idx = -1                                 #last idx that has a value that is nonzero
                        for i in range(num_left):                     #running through all the remaining values
                            interval = self.storage[i][0]-prev_idx    #the distance from this nonzero value to the previous nonzero value
                            #print(i,"  interval: ", interval, "max_interval: ", max_interval, "prev_idx", prev_idx)
                            if interval > max_interval:                        
                                max_interval = interval
                                best_idx = i
                            elif interval == max_interval:                     #if the distance is the same, compare the current position's value with the recorded best one
                                if self.storage[i][1] < self.storage[best_idx][1]: 
                                    best_idx = i
                            prev_idx = self.storage[i][0]                      #previous idx is data[0] of the current data
                       
                        num_left = num_left - 1                       #picked out one to remove, one less to remove in the future

                        #print("best_idx: ", best_idx)
                
                        self.storage = [self.storage[i] if i < best_idx else self.storage[i+1] \
                            for i in range(num_left)]          #update the storage buffer, now you have one less valid elements

                        #print("updated storage: ")
                        #print(self.storage)
                        #print("---------------------------")
                    self.buffer[self.write_to] = [self.storage[i] for i in range(self.num_nonzero)] #after removing all the extra data, store to the write buffer
                    self.storage = [0 for i in range(self.block_size)]    #reset the storage
                self.ready_to_switch = True
                
            #print(self.buffer[self.write_to])

        # switch which buffer to write to
        # (makes it easier to avoid overwiting unread data)
        if (self.ready_to_switch and self.read_from == self.write_to):
            #print("switching......")
            self.write_to = 1-self.write_to
            self.prev_valid_marker = self.valid_marker
            self.valid_marker = 0
            self.ready_to_switch = False


class ThresholdPruner(Module):
    def instantiate(self, in_chn, out_chn, num_nonzero, block_size, preserve_order=True):
        self.name = 'clustered_pruner'

        # num_nonzero is the number of nonzero activations to keep
        # The rest will be pruned blockwise
        self.num_nonzero = num_nonzero


        self.block_size = block_size
        self.threshold = 0.4              #prune out values that are less than 0.4*max first 

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
        # storage stores the whole block of activations for selecting which ones to prune
        self.storage = [0 for i in range(self.block_size)] 
        self.storage_thresholded = [0 for i in range(self.block_size)] 
        self.buffer = [[0 for i in range(self.num_nonzero)] for j in [0,1]]
        self.data_out = [[0 for i in range(3)] for j in range(self.num_nonzero)]
        #self.data_out = [0 for i in range(3*self.num_nonzero)]

        # channels
        self.in_chn = in_chn
        self.out_chn = out_chn

    def tick(self):
        # Send values to output channel from buffer
        if not (self.read_from == self.write_to) and self.out_chn.vacancy():
            data = self.buffer[self.read_from][self.out_marker]
            # prev marker marks how many nonzeros there are in the last round
            if (self.out_marker == self.prev_valid_marker-1):  
                # if reached the last valid activation
                data.append(1) # make sure to add termination bit
                self.data_out[self.out_marker] = data
                self.out_chn.push(self.data_out)
                #print("Pruner out:")
                #print(self.data_out)

                # reset counters, switch to the other buffer
                self.data_out = [[0 for i in range(3)] for j in range(self.num_nonzero)]
                self.out_marker = 0
                self.read_from = 1-self.read_from #toggle the read/write buffer
            else:
                data.append(0)
                self.data_out[self.out_marker] = data
                self.out_marker += 1            

            #self.out_chn.push(data)        
        
        # Load values from input channel into buffer
        # and perform pruning
        if (self.in_chn.valid() and not self.ready_to_switch):
            data = self.in_chn.pop()                    
            #store the data in the storage buffer              
            self.storage[self.valid_marker] = [data[0], data[1]]
            self.valid_marker += 1
            if data[2] == 1:   #if this is the last data of the block, need to prune the block and store it to write-to buffer  

                #print(self.storage)  
                #print("valid_marker: ", self.valid_marker)   
                # s = 0
                # for idx in range(self.valid_marker):
                #     s += self.storage[idx][1]
                # avg = s/self.valid_marker if self.valid_marker!=0 else 0    #get the average value of this block, zero if no nonzero values
                dummy = lambda x: self.storage[x][1] # give value of activation
                x = range(self.valid_marker)
                m = max([self.storage[i][1] for i in range(self.valid_marker)])
                threshold_value = int(m*self.threshold)                   #calculate the threshold value
                #print("-------- max_value: ", m)
                print("-------- threshold_value: ", threshold_value)
                thresholded_idx = 0

                for i in range(self.valid_marker):
                    if self.storage[i][1] >= threshold_value:
                        self.storage_thresholded[thresholded_idx] = self.storage[i]
                        thresholded_idx += 1
                #print("storage_thresholded")
                #print(self.storage_thresholded)

                if thresholded_idx <= self.num_nonzero:   #if there are less than num_nonzero non-zero numbers
                    self.buffer[self.write_to][0:thresholded_idx] = [self.storage_thresholded[i] for i in range(thresholded_idx)]
                    self.valid_marker = thresholded_idx
               
                else: #if there are more than num_nonzero numbers even after thresholding,prune the smallest ones
                    num_remove = thresholded_idx - self.num_nonzero  #number of values that need to be removed
                    num_left = thresholded_idx                       #number of values that are left in the buffer
                    self.valid_marker = self.num_nonzero
                    for r in range(num_remove): 
                        dummy = lambda x: self.storage_thresholded[x][1] # give value of activation
                        x = range(num_left)
                        min_idx = min(x, key=dummy)
                        num_left = num_left - 1
                        self.storage = [self.storage_thresholded[i] if i < min_idx else self.storage_thresholded[i+1] \
                            for i in range(num_left)]          #update the storage buffer, now you have one less valid elements

                        #print("updated storage: ")
                        #print(self.storage)
                        #print("---------------------------")
                    self.buffer[self.write_to] = [self.storage_thresholded[i] for i in range(self.num_nonzero)] #after removing all the extra data, store to the write buffer
                    self.storage = [0 for i in range(self.block_size)]    #reset the storage
                self.ready_to_switch = True
                
            #print(self.buffer[self.write_to])

        # switch which buffer to write to
        # (makes it easier to avoid overwiting unread data)
        if (self.ready_to_switch and self.read_from == self.write_to):
            #print("switching......")
            self.write_to = 1-self.write_to
            self.prev_valid_marker = self.valid_marker
            self.valid_marker = 0
            self.ready_to_switch = False

