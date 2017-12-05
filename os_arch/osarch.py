from nnsim.module import Module, ModuleList
from nnsim.reg import Reg
from nnsim.channel import Channel

from pe import PE
from serdes import InputDeserializer, OutputSerializer
from glb import IFMapGLB, WeightsGLB, PSumGLB, IFMapWeightsGLB
from noc import IFMapNoC, WeightsNoC, PSumRdNoC, PSumWrNoC

class OSArch(Module):
    def instantiate(self, arr_y,
            input_chn, output_chn,
            block_size, num_nonzero,
            ifmap_glb_depth, psum_glb_depth, weight_glb_depth):
        # PE static configuration (immutable)
        self.name = 'chip'
        #self.arr_x = arr_x
        self.arr_y = arr_y
        self.block_size = block_size
        self.num_nonzero = num_nonzero
        
        self.stat_type = 'show'

        # Instantiate DRAM IO channels
        self.input_chn = input_chn
        self.output_chn = output_chn

        # Instantiate input deserializer and output serializer
        self.ifmap_wr_chn = Channel()
        self.psum_wr_chn = Channel()
        self.weights_wr_chn = Channel()
        arr_x = 1
        chn_per_word = 1
        self.deserializer = InputDeserializer(self.input_chn, self.ifmap_wr_chn,
                self.weights_wr_chn, self.psum_wr_chn, arr_y,
                block_size, num_nonzero)

        self.psum_output_chn = Channel()
        self.serializer = OutputSerializer(self.output_chn, self.psum_output_chn)

        # Instantiate GLB and GLB channels
        self.ifmap_rd_chn = Channel(3)
        #self.ifmap_glb = IFMapGLB(self.ifmap_wr_chn, self.ifmap_rd_chn, arr_y,
        #        ifmap_glb_depth, block_size, num_nonzero)

        self.psum_rd_chn = Channel(3)
        self.psum_noc_wr_chn = Channel()
        self.psum_glb = PSumGLB(self.psum_wr_chn, self.psum_noc_wr_chn, self.psum_rd_chn,
                psum_glb_depth, block_size, num_nonzero)

        self.weights_rd_chn = Channel()
        #self.weights_glb = WeightsGLB(self.weights_wr_chn, self.weights_rd_chn, weight_glb_depth, block_size)

        self.ifmap_weights_glb = IFMapWeightsGLB(self.ifmap_wr_chn, self.ifmap_rd_chn,\
            self.weights_wr_chn, self.weights_rd_chn, arr_y, ifmap_glb_depth,\
            weight_glb_depth, block_size, num_nonzero)
        # PE Array and local channel declaration
        self.pe_array = ModuleList()
        self.pe_ifmap_chns = ModuleList()
        self.pe_filter_chns = ModuleList()
        self.pe_psum_in_chns = ModuleList()
        #self.pe_psum_chns_in.append(ModuleList())
        self.pe_psum_out_chns = ModuleList()
        #self.pe_psum_chns_out.append(ModuleList())
        #for x in range(self.arr_x):
        #    self.pe_psum_chns_in[0].append(Channel(32))
        #    self.pe_psum_chns_out[0].append(Channel(32))

        # Actual array instantiation
        for y in range(self.arr_y):
            self.pe_array.append(ModuleList())
            self.pe_ifmap_chns.append(ModuleList())
            self.pe_filter_chns.append(ModuleList())
            self.pe_psum_in_chns.append(ModuleList())
            self.pe_psum_out_chns.append(ModuleList())
            for x in range(1):
                self.pe_ifmap_chns[y].append(Channel(32))
                self.pe_filter_chns[y].append(Channel(32))
                self.pe_psum_in_chns[y].append(Channel(32))
                self.pe_psum_out_chns[y].append(Channel(32))
                self.pe_array[y].append(
                    PE(x, y,
                        self.pe_ifmap_chns[y][x],
                        self.pe_filter_chns[y][x],
                        self.pe_psum_in_chns[y][x],
                        self.pe_psum_out_chns[y][x]
                    )
                )

        # Setup NoC to deliver weights, ifmaps and psums
        self.filter_noc = WeightsNoC(self.weights_rd_chn, self.pe_filter_chns, block_size)
        self.ifmap_noc = IFMapNoC(self.ifmap_rd_chn, self.pe_ifmap_chns)
        self.psum_rd_noc = PSumRdNoC(self.psum_rd_chn, self.pe_psum_in_chns, self.arr_y, block_size)
        self.psum_wr_noc = PSumWrNoC(self.pe_psum_out_chns, self.psum_noc_wr_chn, self.psum_output_chn, self.arr_y, block_size)

    def configure(self, image_size, filter_size, in_chn, out_chn):
        in_sets = in_chn//self.block_size
        out_sets = out_chn//self.arr_y
        fmap_per_iteration = image_size[0]*image_size[1]
        num_iteration = filter_size[0]*filter_size[1]

        self.deserializer.configure(image_size, filter_size, in_chn, out_chn)
        #self.ifmap_glb.configure(image_size, filter_size, in_chn, fmap_per_iteration)
        self.psum_glb.configure(filter_size, out_chn, fmap_per_iteration)
        #self.weights_glb.configure(filter_size, image_size, in_chn, out_chn)
        self.filter_noc.configure(out_chn//self.block_size, self.arr_y, in_chn, out_chn)
        self.ifmap_noc.configure(in_sets, self.arr_y)
        self.psum_rd_noc.configure(out_chn//self.block_size)
        self.psum_wr_noc.configure(fmap_per_iteration, self.arr_y//self.block_size)

        self.ifmap_weights_glb.configure(image_size, filter_size, in_chn, out_chn,\
            fmap_per_iteration)

        for y in range(self.arr_y):
            for x in range(1):
                self.pe_array[y][x].configure(num_iteration*in_sets, fmap_per_iteration*out_sets)
