from nnsim.module import Module

class WeightsNoC(Module):
    def instantiate(self, rd_chn, wr_chns, block_size):
        self.name = 'weight_noc'
        self.block_size = block_size
        
        self.stat_type = 'show'
        self.raw_stats = {'noc_multicast' : 0}

        self.rd_chn = rd_chn
        self.wr_chns = wr_chns

        self.filter_sets = 0
        self.arr_y = 0
        self.arr_x = 0

        self.curr_set = 0
        self.curr_filter = 0

    def configure(self, filter_sets, in_chn, out_chn, arr_y):
        self.filter_sets = filter_sets
        self.arr_y = arr_y
        self.in_chn = in_chn
        self.out_chn = out_chn

        self.curr_set = 0
        self.curr_filter = 0

    def tick(self):
        # Dispatch filters to PE columns. (PE is responsible for pop)
        #print(self.rd_chn.valid())
        if self.rd_chn.valid():
            vacancy = True
            #xmin = self.curr_set*self.chn_per_word
            #xmax = xmin + self.chn_per_word
            ymin = self.curr_set * self.block_size
            ymax = ymin + self.block_size
            #for y in range(ymin, ymax):
            for y in range(ymin, ymax):
                #for x in range(xmin, xmax):
                vacancy = vacancy and self.wr_chns[y][0].vacancy()
            #print(vacancy)
            if vacancy:
                data = self.rd_chn.pop()
                #print("weights_noc")
                #print(data)
                #print(self.curr_filter)
                self.raw_stats['noc_multicast'] += len(data) # counted from the input side
                #for y in range(ymin, ymax):
                for y in range(ymin, ymax):
                    #for x in range(xmin, xmax):
                    #print(data)
                    #print("Weight push {},{}".format(y,x))
                    self.wr_chns[y][0].push(data[y-ymin])

                self.curr_set += 1
                if self.curr_set == self.filter_sets:
                    self.curr_set = 0
                #self.curr_filter += 1
                #if self.curr_filter == self.out_chn:
                #    self.curr_filter = 0

class IFMapNoC(Module):
    def instantiate(self, rd_chn, wr_chns):
        self.name = 'ifmap_noc'
        
        self.stat_type = 'show'
        self.raw_stats = {'noc_multicast' : 0}

        self.rd_chn = rd_chn
        self.wr_chns = wr_chns

        self.ifmap_sets = 0

        self.curr_set = 0
        self.curr_filter = 0

    def configure(self, ifmap_sets, arr_y):
        self.ifmap_sets = ifmap_sets
        
        self.arr_y = arr_y

        self.curr_set = 0
        self.curr_filter = 0

    def tick(self):
        # Feed inputs to the PE array from the GLB
        if self.rd_chn.valid():
            # Dispatch ifmap read if space is available and not at edge
            
            vacancy = True
            #for y in range(self.arr_y):
            for y in range(self.arr_y):
                vacancy = vacancy and self.wr_chns[y][0].vacancy()
            #print(vacancy)
            if vacancy:
                data = self.rd_chn.pop()
                data_val = data[1]
                #print("ifmap_noc")
                #print(data)
                self.raw_stats['noc_multicast'] += len(data)
                #for y in range(ymin, ymax):
                for y in range(self.arr_y):
                    # Send in data[0], the tag that tells us where we are
                    # (in terms of what output we are producing)
                    # and data[1], the actual input data value
                    self.wr_chns[y][0].push([data[0],data[1]])

                #self.curr_set += 1
                #if self.curr_set == self.ifmap_sets:
                #    self.curr_set = 0

# Sends partial values to PEs
class PSumRdNoC(Module):
    def instantiate(self, rd_chn, wr_chns, arr_y, block_size):
        self.name = 'psum_rd_noc'
        self.block_size = block_size
        
        self.stat_type = 'show'
        self.raw_stats = {'noc_multicast' : 0}

        self.arr_y = arr_y
        
        self.rd_chn = rd_chn
        self.wr_chns = wr_chns

        self.psum_sets = 0

        self.curr_set = 0

    def configure(self, psum_sets):
        self.psum_sets = psum_sets

        self.curr_set = 0

    def tick(self):
        # Feed psums to the PE array from the GLB
        if self.rd_chn.valid():
            # Dispatch ifmap read if space is available and not at edge
            ymin = self.curr_set * self.block_size
            ymax = ymin + self.block_size
            vacancy = True
            #for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                vacancy = vacancy and self.wr_chns[y][0].vacancy()

            if vacancy:
                data = self.rd_chn.pop()
                #print("psum_rd_noc")
                #print(data)
                self.raw_stats['noc_multicast'] += len(data)
                #for x in range(xmin, xmax):
                for y in range(ymin, ymax):
                    #print("Bias push {},{}".format(y,x))
                    self.wr_chns[y][0].push(data[y - ymin])

                self.curr_set += 1
                if self.curr_set == self.psum_sets:
                    self.curr_set = 0

# Receives partial values from PEs
class PSumWrNoC(Module):
    def instantiate(self, rd_chns, glb_chn, output_chn, arr_y, block_size):
        self.block_size = block_size
        self.name = 'psum_wr_noc'
        
        self.stat_type = 'show'
        self.raw_stats = {'noc_multicast' : 0}

        self.rd_chns = rd_chns
        self.glb_chn = glb_chn
        self.output_chn = output_chn
        
        self.arr_y = arr_y

        self.num_iteration = 0
        self.psum_sets = 0

        self.iteration = 0
        self.curr_set = 0

    def configure(self, num_iteration,
            psum_sets):
        self.num_iteration = num_iteration
        self.psum_sets = psum_sets

        self.iteration = 0
        self.curr_set = 0

    def tick(self):
        # Check if psum available for write-back
        valid = True
        ymin = self.curr_set*self.block_size
        ymax = ymin + self.block_size
        #print(xmax)
        #print(self.curr_set)
        #print(self.rd_chns[self.psum_idx][xmax].valid())
        for y in range(ymin, ymax):
            valid = valid and self.rd_chns[y][0].valid()

        
        if valid:
            #print("valid")
            target_chn = self.output_chn# if self.iteration == \ # Due to accumulating locally, should never send to GLB
                    #self.num_iteration-1 else self.glb_chn
            if target_chn.vacancy():
                data = [ self.rd_chns[y][0].pop() for y in range(ymin, ymax) ]
                self.raw_stats['noc_multicast'] += len(data) # count all data points from the input side
                target_chn.push(data)
                #print("psum_wr_noc")
                #print(data)

                self.curr_set += 1
                if self.curr_set == self.psum_sets:
                    self.curr_set = 0
                    self.iteration += 1
