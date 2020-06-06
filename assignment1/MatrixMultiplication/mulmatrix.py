
# param 1: file name 
# param 2-N: different block sizes

import sys
import subprocess
import matplotlib.pyplot as plt
import matplotlib.cm as cm

file_name = sys.argv[1]
block_sizes = map(lambda block_size: int(block_size), sys.argv[2:])
methods = ['naive','blocked']
subprocess.call(["gcc", "matrix_multiplication.c"])

print("This script executes matrix multiplication for the matrixes in file",
        "'" + file_name + "'", "using the naive strategy and the blocked",
        "strategy for block sizes:", ", ".join(sys.argv[2:]))

naive = []
blocked = []
f = open(file_name, "r")
lines = f.readlines()
f.close()
for line in lines:
	for method in methods:
		if method == 'naive':
			arg_naive = "./matrix_multiplication" + ' ' + method + ' ' + line.strip('\n')
			p1 = subprocess.Popen(arg_naive.split(), stdout=subprocess.PIPE)
			out1 = p1.communicate()[0]
			naive.append([int(line.split()[0]), float(out1.split(' ')[1].strip('\n'))])
		else:
			for block in block_sizes:
				arg_blocked = "./matrix_multiplication" + ' ' + method + ' ' + line.strip('\n') + ' ' + str(block)
				p2 = subprocess.Popen(arg_blocked.split(), stdout=subprocess.PIPE)
				out2 = p2.communicate()[0]
				blocked.append([int(line.split()[0]), block, float(out2.split(' ')[1].strip('\n'))])
		
print(naive,blocked)

print("The result of this script is a graph 'graph.png' in the current",
      "directory that shows the results with matrix size on the x-axis and",
      "time on the y-axis. Create one line for each block size and one line",
      "for the naive strategy.")

for block in block_sizes:
	time=[]
	inp=[]
	for item in blocked:
		if item[1] == block:
			time.append(item[2])
			inp.append(item[0])
	plt.plot(inp, time)

for item in naive:
	plt.axhline(item[1], linestyle = '--', color = 'y')

plt.xlabel('matrix_size')
plt.ylabel('time_spent')
plt.legend([str(block) for block in block_sizes]+['naive'+str(item[0]) for item in naive], loc = 'upper left')
plt.title('Naive vs Blocked Matrix Multiplication')
plt.savefig('graph.png')
plt.show()