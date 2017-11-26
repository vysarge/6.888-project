from random import randint

from nnsim.module import Module
from nnsim.reg import Reg
from nnsim.channel import Channel
from nnsim.simulator import Finish

from converter import Converter

class ConverterTB(Module):
    def instantiate(self):
        self.name = 'tb'

        self.input_size = 4
        self.block_size = 12
        self.in_sets = self.block_size // self.input_size

        self.in_chn = Channel()
        self.out_chn = Channel()

        self.dut = Converter(self.in_chn, self.out_chn, self.input_size, self.block_size)

        self.iterations = 10
        self.iteration = 0
        self.curr_set = 0
        self.out_counter = 0
        self.test_data = [[randint(1,5) if randint(0,3)>2 else 0\
            for j in range(self.block_size)]\
            for i in range(self.iterations+1)] 
            # send in one extra iteration to flush out last outputs
        print("Stimulus:")
        print("[")
        for i in range(len(self.test_data)):
            print(self.test_data[i])
        print("]")

    def tick(self):
        if (self.in_chn.vacancy() and not self.iteration == self.iterations+1):
            imin = self.curr_set*self.input_size
            imax = imin+self.input_size
            data = [self.test_data[self.iteration][i] for i in range(imin, imax)]
            self.in_chn.push(data)

            self.curr_set += 1
            if (self.curr_set == self.in_sets):
                self.curr_set = 0
                self.iteration += 1
        if (self.out_chn.valid()):
            data = self.out_chn.pop()
            print(data)
            if (data[2] == 1):
                self.out_counter += 1
            if (self.out_counter == self.iterations):
                raise Finish("Check manually")

if __name__ == "__main__":
    from nnsim.simulator import run_tb
    converter_tb = ConverterTB()
    run_tb(converter_tb, verbose=False)
