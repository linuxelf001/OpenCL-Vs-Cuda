import os, subprocess
from timeit import Timer

def callCUDA():
#	change qd-newCUDA to the correct name of the executable which runs the algorithm using CUDA
	subprocess.call("./cuda", 0, None, None, None, None)
	
def callOpenCL():
#	change qd-newCPU to the correct name of the executable which runs the algorithm using CPU
	subprocess.call("./qd", 0, None, None, None, None)

if __name__=='__main__':
	
	f = open('performanceResult', 'w')
	f.write('Mutliprecision library - CPU/CUDA comparison\n')

	time = min(Timer("callCUDA()","from __main__ import callCUDA").repeat(5,1))
	mystr = 'Time needed using CUDA:\n' + str(time) +'\n'
	f.write(mystr)
	
	f.write('\n\nFloating point vector addition using OpenCL\n')

	time = min(Timer("callOpenCL()","from __main__ import callOpenCL").repeat(5,1))
	mystr = 'Time needed using CPU:\n' + str(time) +'\n'
	f.write(mystr)

	f.close()
