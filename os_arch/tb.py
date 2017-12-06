from nnsim.module import Module
from nnsim.channel import Channel
from osarch import OSArch
from stimulus import Stimulus

debug = True # allows switching between a main testing conv and a debug one
debugStimulus = False # if true, uses integer and increasing values for inputs, weights, and biases
keepMaxValues = True # if true, always keeps num_nonzero values in each block
do_premature_prune = True # if true, will pre-prune inputs to allow output deserializer to do correct validation
# For actual test runs, should be False, False, True, False

class OSArchTB(Module):
    def instantiate(self):
        self.name = 'tb'
        
        if (debug):
            self.image_size = (4, 4)
            self.filter_size = (3, 3)
            self.in_chn = 2
            self.out_chn = 4
            self.block_size = 2
            self.num_nonzero = 1
        else:
            self.image_size = (16, 16)
            self.filter_size = (3, 3)
            self.in_chn = 16
            self.out_chn = 8
            self.block_size = 4
            self.num_nonzero = 4

        self.arr_y = self.out_chn

        self.input_chn = Channel()
        self.output_chn = Channel()

        ifmap_glb_depth = (self.filter_size[1] + (self.filter_size[0]-1)*\
            self.image_size[1]) * self.in_chn // self.block_size
        psum_glb_depth = self.out_chn // self.block_size
        weight_glb_depth = self.filter_size[0]*self.filter_size[1]* \
                self.in_chn*self.out_chn//self.block_size

        self.stimulus = Stimulus(self.arr_y, self.block_size, self.num_nonzero,
            self.input_chn, self.output_chn)
        self.dut = OSArch(self.arr_y, self.input_chn, self.output_chn, self.block_size,
            self.num_nonzero, ifmap_glb_depth, psum_glb_depth, weight_glb_depth)

        self.configuration_done = False

    def tick(self):
        if not self.configuration_done:
            self.stimulus.configure(self.image_size, self.filter_size, self.in_chn, self.out_chn, debugStimulus, keepMaxValues, do_premature_prune)
            self.dut.configure(self.image_size, self.filter_size, self.in_chn, self.out_chn)
            self.configuration_done = True


if __name__ == "__main__":
    from nnsim.simulator import run_tb
    os_tb = OSArchTB()
    run_tb(os_tb, verbose=False)
