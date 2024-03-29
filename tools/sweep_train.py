from os.path import exists
from os import chdir, getcwd, makedirs
from os import path
import subprocess
import fileinput
import random
import sys
import time
import wandb


## define global variables
project='athena_mnist_test'
codename='athena_test'
sweep_id = 'vw5xt1zq' #'hfyfmd71' #'8336mo4r' #'emjb3nsc' #'8qcw0m55'
count=100
file_template = "template_momentum.in"
script_directory = path.dirname(path.abspath(sys.argv[0]))
workdir = getcwd() # "../"
min_neurons = 12
max_neurons = 100

## check whether template file exists
## ... if not, exit
if not exists(file_template):
    f = open(file_template,"x")
    # write out example file
    print("ERROR: "+file_template+" NOT PRESENT")
    print("Exiting...")
    f.close()
    sys.exit()


## make sweep directory and cd
sweepdir = workdir+"/D_"+sweep_id
makedirs(sweepdir, exist_ok=True)
chdir(sweepdir)


def main():

    #run = wandb.init(project=project)

    ## initialise run
    wandb.init()
    agent_id = wandb.run.id
    agent_name = wandb.run.name

    ## set multiple metrics
    wandb.define_metric("loss",summary="min")
    wandb.define_metric("accuracy",summary="max")

    ## set up logical dictionary for file card editing
    l_dict = {
        'setup': False,
        'training': False,
        'convolution': False,
        'pooling': False,
        'fully_connected': False
    }
    
    ## set up parameter dictionary for file editing
    param_dict = {
        'setup': {},
        'training': {},
        'convolution': {},
        'pooling': {},
        'fully_connected': {}
    }

    hidden_layers = [random.randint(min_neurons,max_neurons) 
                     for _ in range(wandb.config.num_hidden_layers)]
    wandb.config.update({"hidden_layers":hidden_layers})
    
    ## populate parameter dictionary
    param_dict['training'] = {
        'learning_rate': wandb.config.learning_rate,
        #'l1_lambda': wandb.config.l1_lambda,
        #'l2_lambda': wandb.config.l2_lambda,
        'batch_size': wandb.config.batch_size,
        'num_epochs': wandb.config.num_epochs,
        'momentum': wandb.config.momentum
    }
    param_dict['convolution'] = {
        'cv_num_filters': wandb.config.cv_num_filters,
        'clip_norm': wandb.config.cv_clip_norm,
        'kernel_size': wandb.config.cv_kernel_size,
        'stride':wandb.config.cv_stride
    }
    param_dict['pooling'] = {
        'kernel_size': wandb.config.pool_kernel_size,
        'stride':wandb.config.pool_stride
    }
    param_dict['fully_connected'] = {
        'clip_norm': wandb.config.fc_clip_norm,
        'hidden_layers': "'"+", ".join(
            str(num) for num in wandb.config.hidden_layers)+"'" #wandb.config.hidden_layers
    }

    ## open template input file and save to string
    with open(workdir+"/"+file_template, 'r') as file:
        newline=[]
        for line in file.readlines():
            if '&' in line:
                card = line.replace('&','').strip()
                for key in l_dict.keys():
                    if card == key:
                        l_dict[key] = True
                        break
            elif line.strip() == '/':
                l_dict = dict.fromkeys(l_dict,False)

            edited = False
            true_keys = [key for key, value in l_dict.items() if value == True]

            if not true_keys:
                newline.append(line)
                continue

            if line.strip().endswith(','):
                end = ','
            else:
                end = ''
                
            for key, value in param_dict[true_keys[0]].items():
                #edited = update_tag(line, key, value,end)
                #if key in line:
                tag = line.split('=')[0]
                if key in tag:
                    newline.append(tag+"= "+str(value)+end+'\n')
                    edited = True
                    if edited:
                        break
            if not edited:
                newline.append(line)
                
    ## make agent directory and cd
    curdir = sweepdir+"/D_"+agent_name
    makedirs(curdir, exist_ok=True)
    chdir(curdir)

    ## make parameter file for run
    file_param = "param.in"
    with open(file_param,"w") as file:
        for line in newline:
            file.writelines(line)

    ## set output file names and run fortran executable
    stdout = "stdout.o"
    stderr = "stdout.e"
    stdout_file = open(stdout, 'w')
    stderr_file = open(stderr, 'w')
    p = subprocess.Popen(args=[script_directory+"../test/bin/"+codename,
                             "-f"+file_param],
                       stdout=stdout_file,
                       stderr=stderr_file
    )
    #exit_code = p.wait()

    ## read from the output file and log to wandb
    ## ... do this interactively with the fortran code running
    ## ... check every 1 second for an updated line
    lastLine = None
    index = -1
    #time.sleep(5)
    while p.poll() is None:
        with open(stdout,'r') as f:
            lines = f.readlines()
            if lines is None or not lines:
                time.sleep(5)
                continue
            if lastLine is None:
                lastLine = lines[0]
                index = -1
            #else:
            #    index = lines.index(lastLine,start=index)
            if lines[-1] != lastLine:
                for i in range(index+1,len(lines)):
                    if 'epoch=' in lines[i]:
                        result_dict = {}
                        pairs = lines[i].split(",")
                        for pair in pairs:
                            key, value = pair.split("=")
                            if key == 'epoch' or key == 'batch':
                                result_dict[key.strip()] = int(value.strip())
                            else:
                                result_dict[key.strip()] = float(value.strip())
                        wandb.log(result_dict)
                        #wandb.log({"train": result_dict})
                    elif 'Overall accuracy' in lines[i]:
                        key, value = lines[i].split("=")
                        wandb.run.summary["test accuracy"] = value                        
                    index = i
                lastLine = lines[-1]
            time.sleep(5)

    ## close stdout and stderr files
    stdout_file.close()
    stderr_file.close()
    
    ## read from output file and log to wandb
    #with open(stdout,'r') as file:
    #    for line in file.readlines():
    #        if 'epoch' in line:
    #            ##epoch=1, batch=1, lrate=.010, error=67.023
    #            result_dict = {}
    #            pairs = line.split(",")
    #            for pair in pairs:
    #                key, value = pair.split("=")
    #                result_dict[key.strip()] = value.strip()
    #            wandb.log(result_dict)

    ## return to parent directory
    chdir(sweepdir)


## set up sweep agents
print("Logging in...")
wandb.login()
#run = wandb.init(project=project)
print("Setting off agents")
wandb.agent(sweep_id=sweep_id, function=main, count=count, project=project)
print("All agents complete")


wandb.finish()
