import os
import subprocess

#Config
outfile = "results.txt"
outfolder = "profiling_new"
cppfolder = "../../adaptation_oscillations_standalone_cpp/adaptation_oscillations_standalone_cpp/"

#values_num_blocks = [0.5, 1, 2, 4]	#not real number, but multiplier
values_num_blocks = [0.5, 1, 2, 4]
#values_N_neurons = [1000, 4000, 20000, 100000]
values_N_neurons = [4000]
#values_sparsity = [0.01, 0.03, 0.05, 0.08, 0.1, 0.2]
values_sparsity = []
#values_input_mean = [0.07, 0.14, 0.21, 0.28, 0.56]
values_input_mean = []
#values_runtime = [0.1, 0.5, 1.0, 1.5, 2.0]
values_runtime = []

normal_num_blocks = 1
normal_N_neurons = 4000
normal_sparsity = 0.05
normal_input_mean = 0.14
normal_runtime = 1.0

def getFileName(num_blocks=normal_num_blocks, N_neurons=normal_N_neurons, sparsity=normal_sparsity, input_mean=normal_input_mean, runtime=normal_runtime):
	fileName = outfolder + "/" + str(num_blocks) + "_" + str(N_neurons) + "_" + str(sparsity) + "_" + str(input_mean) + "_" + str(runtime) + ".txt"
	print "Profiling", fileName
	return fileName

def change_num_blocks(old_value, new_value):
	old_string = "num_blocks = props.multiProcessorCount * " + str(old_value) + ";"
	new_string = "num_blocks = props.multiProcessorCount * " + str(new_value) + ";"
	os.system("sed -i \"s/" + old_string +"/"+ new_string + "/\" objects.cu")

def change_N_neurons(old_value, new_value):
	old_string = "unsigned int brian::neurongroup_N = " + str(old_value) + ";"
	new_string = "unsigned int brian::neurongroup_N = " + str(new_value) + ";"
	os.system("sed -i \"s/" + old_string +"/"+ new_string + "/\" objects.cu")
	os.system("sed -i \"s/" + str(old_value) +"/"+ str(new_value) + "/\" " + cppfolder + "*.cpp")
	os.system("sed -i \"s/" + str(old_value + 1) +"/"+ str(new_value + 1) + "/\" " + cppfolder + "*.cpp")
	os.system("sed -i \"s/" + str(old_value) +"/"+ str(new_value) + "/\" " + cppfolder + "*/*.cpp")
	os.system("sed -i \"s/" + str(old_value + 1) +"/"+ str(new_value + 1) + "/\" " + cppfolder + "*/*.cpp")
	os.system("rm -f static_arrays/_static_array__array_neurongroup*")
	os.system("rm -f " + cppfolder + "static_arrays/_static_array__array_neurongroup*")
	content = "\x01" * new_value
	f = open("static_arrays/_static_array__array_neurongroup_not_refractory", "w")
	f.write(content)
	f.close()
	f = open(cppfolder + "static_arrays/_static_array__array_neurongroup_not_refractory", "w")
	f.write(content)
	f.close()
	content = ("\x00" * 6 + "\xf0\xff") * new_value
	f = open("static_arrays/_static_array__array_neurongroup_lastspike", "w")
	f.write(content)
	f.close()
	f = open(cppfolder + "/static_arrays/_static_array__array_neurongroup_lastspike", "w")
	f.write(content)
	f.close()

def change_sparsity(old_value, new_value):
	old_string = "const double _p = " + str(old_value) + ";"
	new_string = "const double _p = " + str(new_value) + ";"
	os.system("sed -i \"s/" + old_string +"/"+ new_string + "/\" code_objects/synapses_synapses_create_codeobject.cu")
	os.system("sed -i \"s/" + old_string +"/"+ new_string + "/\" " + cppfolder + "/code_objects/synapses_synapses_create_codeobject.cpp")

def change_input_mean(old_value, new_value):
	old_string = "const double _v = _dt * (" + str(old_value) + " * int_(not_refractory) - v * int_(not_refractory) / 0.01 - w * int_(not_refractory) / 0.01) + v + 0.002213594362117866 * xi * int_(not_refractory);"
	new_string = "const double _v = _dt * (" + str(new_value) + " * int_(not_refractory) - v * int_(not_refractory) / 0.01 - w * int_(not_refractory) / 0.01) + v + 0.002213594362117866 * xi * int_(not_refractory);"
	os.system("sed -i \"s/" + old_string +"/"+ new_string + "/\" code_objects/neurongroup_stateupdater_codeobject.cu")
	os.system("sed -i \"s/" + old_string +"/"+ new_string + "/\" " + cppfolder + "code_objects/neurongroup_stateupdater_codeobject.cpp")

def change_runtime(old_value, new_value):
	old_string = "magicnetwork.run(" + str(old_value) + ", NULL, 10.0);"
	new_string = "magicnetwork.run(" + str(new_value) + ", NULL, 10.0);"
	os.system("sed -i \"s/" + old_string +"/"+ new_string + "/\" main.cu")
	os.system("sed -i \"s/" + old_string +"/"+ new_string + "/\" " + cppfolder + "main.cpp")

def profile(target = "gpu"):
	init_time = 0.0
	sim_time = 0.0
	num_spikes = 0
	out = ""
	if target == "gpu":
		os.system("make")
		p = subprocess.Popen(['./main'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = p.communicate()
	else:
		os.system("cd " + cppfolder + " && make")
		p = subprocess.Popen([cppfolder + "main"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = p.communicate()
	out = out.split("\n")
	print "Out:",out
	print "Err:",err
	for line in out:
		if "Initialization time: " in line:
			init_time = float(line.split(":")[1])
		if "Number of spikes: " in line:
			num_spikes = int(line.split(":")[1])
		if "Simulation time: " in line:
			sim_time = float(line.split(":")[1])
		if "ERROR" in line:
			return "ERROR", "ERROR", "ERROR"
	sim_time = sim_time - init_time
	return init_time, sim_time, num_spikes

def nvprofile(out = "normal.txt"):
	os.system("nvprof ./main >" + out + " 2>&1")

def run_tests():
	for j in values_num_blocks:
		change_num_blocks(normal_num_blocks, j)
		f = open(outfolder + "/" + outfile, "w")
		f.write("num_blocks = " + str(j) + "\n\n\n")
		f.write("neurongroup_N\tGPU (init)\tGPU (sim)\tCPU (sim)\tspikes\n")
		for i in values_N_neurons: 
			change_N_neurons(normal_N_neurons, i)
			f.write(str(i) + "\t")
			init_time, sim_time, num_spikes = profile("gpu")
			f.write(str(init_time) + "\t" + str(sim_time) + "\t")
			init_time, sim_time, num_spikes = profile("cpu")
			f.write(str(sim_time) + "\t" + str(num_spikes) + "\n")
			nvprofile(getFileName(num_blocks=j, N_neurons=i))
			change_N_neurons(i, normal_N_neurons)
			f.flush()
		f.write("sparsity\tGPU (init)\tGPU (sim)\tCPU (sim)\tspikes\n")
		for i in values_sparsity:
			change_sparsity(normal_sparsity, i)
			f.write(str(i) + "\t")
			init_time, sim_time, num_spikes = profile()
			f.write(str(init_time) + "\t" + str(sim_time) + "\t")
			init_time, sim_time, num_spikes = profile("cpu")
			f.write(str(sim_time) + "\t" + str(num_spikes) + "\n")
			nvprofile(getFileName(num_blocks=j, sparsity=i))
			change_sparsity(i, normal_sparsity)
			f.flush()
		f.write("input_mean\tGPU (init)\tGPU (sim)\tCPU (sim)\tspikes\n")
		for i in values_input_mean:
			change_input_mean(normal_input_mean, i)
			f.write(str(i) + "\t")
			init_time, sim_time, num_spikes = profile()
			f.write(str(init_time) + "\t" + str(sim_time) + "\t")
			init_time, sim_time, num_spikes = profile("cpu")
			f.write(str(sim_time) + "\t" + str(num_spikes)+ "\n")
			nvprofile(getFileName(num_blocks=j, input_mean=i))
			change_input_mean(i, normal_input_mean)
			f.flush()
		f.write("runtime\tGPU (init)\tGPU (sim)\tCPU (sim)\tspikes\n")
		for i in values_runtime:
			change_runtime(normal_runtime, i)
			f.write(str(i) + "\t")
			init_time, sim_time, num_spikes = profile()
			f.write(str(init_time) + "\t" + str(sim_time) + "\t")
			init_time, sim_time, num_spikes = profile("cpu")
			f.write(str(sim_time) + "\t" + str(num_spikes) + "\n")
			nvprofile(getFileName(num_blocks=j, runtime=i))
			change_runtime(i, normal_runtime)
			f.flush()
		f.close()
		change_num_blocks(j, normal_num_blocks)

run_tests()