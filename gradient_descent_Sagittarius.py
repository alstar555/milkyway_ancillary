"""
run instructions: 
python3 gradient_descent_Sagittarius.py
nohup python3 gradient_descent_Sagittarius.py & 


adjustabe vars: pwd, parameters, MAX_ITERS, TOLERANCE, LEARNING_RATE, LIKELIHOOD

ssh loshaa@draco.phys.rpi.edu
cnprbrty

"""


#Import necessary packages
import numpy as np
import math
import matplotlib.pyplot as plt 
import os, shutil
import os.path as pth
import subprocess
import copy
import time
from datetime import date, datetime
from sep_lib import *
import multiprocessing
import random


#GLOBAL VARS
TESTING = False
DEBUG = False
show_plot = False
start_time = time.time()
n_streams = 4

MAX_ITERS = 10

ITERS = 0
RUNTIME = 0
TOTAL_NBODY_RUNS = 0
ROOT = -999


#path 
pwd = '/home/loshaa'
# pwd = '/home/angelica/Documents/Angelica/Rpi/Semesters/5_2022Fall/MilkyWay@Home'
base_path = '/home/loshaa/parsweep/'
separation_binary =  '/home/loshaa/milkyway_separation'
pf_in = '/home/loshaa/stripe81_4s_default.params' #store the format for separation inputs
sf_in = '/home/loshaa/stripe81_4s.txt' # opt by MW@home
star_fn = '/home/loshaa/stars-81_2.txt' # raw SDSS data used for the separation calculations.
test_in = '/home/loshaa/test_params.txt'



PARAMETERS = parse_input_params(sf_in)
NUM_PARAMETERS = len(PARAMETERS)
TOLERANCE = 1e-3
RESET_LEARNING_RATE = 0.7
LEARNING_RATE = RESET_LEARNING_RATE
DECAY = 1.1
GRADIENT = [PARAMETERS[j] * LEARNING_RATE for j in range(NUM_PARAMETERS)] 
# DELTA = [1e-7, 0.5] + [0.5, 1.5, 1.5, 1.2, 1.2, 1.2]*n_streams
# DELTA =[1e-7, 0.5, 
# 0.01, 0.2, 0.8, 0.02, 0.02, 0.04,
# 0.01, 0.2, 0.8, 0.02, 0.02, 0.04,
# 0.01, 0.2, 0.8, 0.02, 0.02, 0.04,
# 0.01, 0.2, 0.8, 0.02, 0.02, 0.2]

#Hiroka's delta
DELTA = [0.00002, 0.005, 
		0.01, 0.2, 0.8, 0.02, 0.02, 0.04,
		0.01, 0.2, 0.8, 0.02, 0.02, 0.04,
		0.01, 0.2, 0.8, 0.02, 0.02, 0.04,
		0.01, 0.2, 0.8, 0.02, 0.02, 0.2]

# gradient: [-0.46420186605722336, -0.03338179710226097, 
# -0.022986563511111946, -0.0006813111909013496, 0.000916110652631567, -0.00455848746948669, -0.007363526503351624, 0.0006612917588751562,
#  -0.0008875241612136264, 2.3308720675426407e-05, 8.476728747301247e-05, -3.101518713213538e-05, 8.412340851852004e-06, -0.0008970190234972956,
#   -0.0008307073729469927, 1.3264884336320708e-05, 0.00015671841030573397, -0.00014509453303452384, 0.0008569011573371961, -0.004109441036500449, 
#   -0.010246615574896367, -3.5218784292245786e-05, 6.130421570974913e-05, 0.0008446807562882223, 0.0009427012098549095, -3.840332769073424e-05]


#scp grad_descent.py loshaa@leot.phys.rpi.edu:
if TESTING:
	pwd = '/home/angelica/Documents/Angelica/Rpi/Semesters/5_2022Fall/MilkyWay@Home'

def remove_file(fname):
	if os.path.exists(fname):
		os.remove(fname)

def keep_in_bounds(p, i):
	up_bounds = [0.99999999999, 1.75] + [999, 400, 60, np.pi, np.pi, 999]*n_streams #upper bounds for each parameter
	low_bounds = [0.001, 0.25] + [-999, 350, 2, 0, -np.pi, 0.001]*n_streams #lower bounds for each parameter

	return max(min(p, up_bounds[i]), low_bounds[i])


def update_parameters(p, n):
	#if n==ROOT, stay in current dir, dont do parallel processes
	if(n==ROOT):
		current_params_fn = 'params.txt'
		remove_file('/results.txt')
	else:
		path = base_path+str(n)
		current_params_fn = path + '/params.txt'
		results_path = path + '/results.txt'
		# Create a new directory if it does not exist
		if not os.path.exists(path):
			 os.makedirs(path)
		else:
			#clean directory if it does exist
			remove_file(current_params_fn)
			remove_file(results_path)

		shutil.copy(base_path+'/params.txt', path)
	update_params_file(pf_in, current_params_fn, p, n_streams)

def run_n_body(p, n):
	global TOTAL_NBODY_RUNS 
	TOTAL_NBODY_RUNS += 1
	#if n==ROOT, stay in current dir, dont do parallel processes
	if(n!=ROOT):
		os.chdir(base_path + str(n) + '/')
	std = subprocess.run([separation_binary, '-a', 'params.txt', '-s', star_fn, '-f', '-t', '-i', '-d', str(n%2)
	])
	#read likelihood
	results_txt = 'results.txt'
	with open(results_txt, 'r') as file:
		return float(file.readline())

#find gradient in parallel process
def get_gradient(i):
	global PARAMETERS
	global DELTA
	p = copy.deepcopy(PARAMETERS)
	# print("parameter", i, ":", PARAMETERS[i], "DELTA:", DELTA[i])
	#change one parameter, hold all others constant
	p[i] = keep_in_bounds(PARAMETERS[i] + DELTA[i], i)
	update_parameters(p, i)
	likelihood1 = run_n_body(p, i) 
	# print("p", i, ":", p[i], "likelihood1:", likelihood1, end=", ")

	p[i] = keep_in_bounds(PARAMETERS[i] - DELTA[i], i)
	update_parameters(p, i)
	likelihood2 = run_n_body(p, i)
	# print("p", -i, ":", p[i], "likelihood2:", likelihood2)
	#find gradient by calc rise/run in likelihoods 
	g = (likelihood1 - likelihood2) / (2 * DELTA[i]) 
	return i, g

def get_step_size(gradient):
	#init big step size
	global PARAMETERS
	global LIKELIHOOD
	global GRADIENT
	global LEARNING_RATE
	global DECAY
	#add some randomness into the parameters
	# if ITERS == 10:
	# 	noise = random.random()
	# 	PARAMETERS = [PARAMETERS[j] + noise for j in range(NUM_PARAMETERS)] 
	# info_file.write(txt)
	# i = 0
	# while True:
	# p = [keep_in_bounds(PARAMETERS[j] + LEARNING_RATE * GRADIENT[j], j) for j in range(NUM_PARAMETERS)] 
	# update_parameters(p, ROOT)
	# likelihood2 = run_n_body(p, ROOT)

	p = [keep_in_bounds(PARAMETERS[j] + LEARNING_RATE * GRADIENT[j], j) for j in range(NUM_PARAMETERS)] 
	update_parameters(p, ROOT)
	likelihood2 = run_n_body(p, ROOT)

	# txt = "likelihood:{}\n ITERS:{}\n\t parameters:{}\n\t LEARNING_RATE:{}\n\t GRADIENT:{}".format(likelihood2, ITERS, p, LEARNING_RATE, GRADIENT)
	# i+=1
	#if worse likelihood
	# if likelihood2 >= LIKELIHOOD: 
	# 	LEARNING_RATE /= 2
	# 	if LEARNING_RATE < TOLERANCE: 
	# 		break
	# else: #if better likelihood, increase step
	LIKELIHOOD = likelihood2
	PARAMETERS = p
	txt = "ITER:{}\t likelihood:{} parameters:{} LEARNING_RATE:{} GRADIENT:{}\n".format(ITERS, likelihood2, p, LEARNING_RATE, GRADIENT)
	info_file.write(txt)
	info_file.flush()
	# SLEARNING_RATETEP *= 3
	LEARNING_RATE /= DECAY
		# break
				
	

def gradient_descent():
	global ITERS
	global RUNTIME
	global LIKELIHOOD
	global START_LIKELIHOOD
	global PARAMETERS
	results = ""
	x = []
	y = []

	#shift all parameters slightly to the left
	# PARAMETERS = [keep_in_bounds(PARAMETERS[j] - TOLERANCE, j) for j in range(NUM_PARAMETERS)] 
	#shift all parameters slightly to the right
	# PARAMETERS = [keep_in_bounds(PARAMETERS[j] + TOLERANCE, j) for j in range(NUM_PARAMETERS)] 


	#init likelihood
	update_parameters(PARAMETERS, ROOT)
	LIKELIHOOD = run_n_body(PARAMETERS, ROOT)
	START_LIKELIHOOD = LIKELIHOOD

	pre_likelihood = LIKELIHOOD + 1

	results += "iter: {}\n".format(ITERS)
	results += "parameters: {}\n".format(PARAMETERS)
	results += "STARTING LIKELIHOOD: {}\n\n".format(LIKELIHOOD)

	while ITERS < MAX_ITERS:
		if(ITERS%10==0):
			LEARNING_RATE = RESET_LEARNING_RATE

		pre_likelihood = LIKELIHOOD
				
        #get gradient in parallel processing
		with multiprocessing.Pool(NUM_PARAMETERS) as pool:
			gradient_tuples = pool.map(get_gradient, range(NUM_PARAMETERS))
		#unpack and store results of gradient
		for g in gradient_tuples:
			GRADIENT[g[0]] = g[1]

		#step in direction of gradient; update likelihood
		get_step_size(GRADIENT)
		info_file.flush()

		#save results to file
		x.append(ITERS)
		y.append(LIKELIHOOD)
		ITERS += 1
		results += "iter: {}\n".format(ITERS)
		results += "parameters: {}\n".format(PARAMETERS)
		results += "LEARNING_RATE: {}\n".format(LEARNING_RATE)
		results += "gradient: {}\n".format(GRADIENT)
		results += "likelihood: {}\n\n".format(LIKELIHOOD)


	results += "total nbody runs: {}\n".format(TOTAL_NBODY_RUNS)
	RUNTIME = (time.time() - start_time)
	results += "total run time: {}\n".format(RUNTIME)
	info_file.write(results)
	info_file.flush()

	if show_plot:
		plt.xlabel("iterations")
		plt.ylabel("likelihood")
		plt.scatter(x, y)
		plt.savefig('likelihood_plot.png')


if __name__ == "__main__":
	print("""STARTING GRAD DESCENT ... version 5\n
		description: 
		step all parameters at same time,
		take gradeint, each parameter has delta[i],
		step parameters scaled based on gradient*learning rate\n""")
		
	info_file = open("more_optimization_info.txt", "a")

	txt = "\n---------------------------------------------\n" + str(date.today()) + " " + str(datetime.now()) + "\n"
	txt += "\nTOLERANCE:{}\nLEARNING_RATE:{}\nDECAY:{}\nDELTA:{}\n\n".format(TOLERANCE, LEARNING_RATE,DECAY,DELTA)
	txt += "note: Parameters shifted slightly to the right by {}\n\n".format(TOLERANCE)
	info_file.write(txt)

	gradient_descent()
	txt = "\nFINAL PARAMETERS:{}\n\nSTART_LIKELIHOOD:{}\n\nRESULT_LIKELIHOOD:{}\niters:{}\nnbody runs:{} \nruntime:{}\nimprovement:{}".format(PARAMETERS, START_LIKELIHOOD,LIKELIHOOD, ITERS,TOTAL_NBODY_RUNS,RUNTIME,(LIKELIHOOD-START_LIKELIHOOD))
	info_file.write(txt)
	info_file.close()

	print(txt)
