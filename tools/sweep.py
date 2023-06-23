from os.path import exists
from os import chdir, getcwd, makedirs
import subprocess
import fileinput
import sys
import wandb

## define global variables
sweep_id = '8qcw0m55'
file_template = "template.in"
curdir = getcwd() # "../"

## check whether template file exists
## ... if not, exit
if not exists(file_template):
    f = open(file_template,"x")
    # write out example file
    print("ERROR: "+file_template+" NOT PRESENT")
    print("Exiting...")
    f.close()
    sys.exit()


def main():

    #run = wandb.init(project='cnn_mnist_test')

    ## initialise run
    wandb.init()

    ## set up logical dictionary for file card editing
    l_dict = {
        'setup': False,
        'training': False,
        'convolution': False,
        'pooling': False,
        'fullyconnected': False
    }
    
    ## set up parameter dictionary for file editing
    param_dict = {
        'setup': {},
        'training': {},
        'convolution': {},
        'pooling': {},
        'fullyconnected': {}
    }

    ## populate parameter dictionary
    param_dict['training'] = {
        'learning_rate': wandb.config.learning_rate,
        'momentum': wandb.config.momentum,
        'l1_lambda': wandb.config.l1_lambda,
        'l2_lambda': wandb.config.l2_lambda,
        'batch_size': wandb.config.batch_size,
        'num_epochs': wandb.config.num_epochs
    }
    param_dict['convolution'] = {
        'cv_num_filters': wandb.config.cv_num_filters,
        'clip_norm': wandb.config.cv_clip_norm
    }
    param_dict['fullyconnected'] = {
        'clip_norm': wandb.config.fc_clip_norm,
        'hidden_layers': wandb.config.hidden_layers
    }

    ## open template input file and save to string
    with open(file_template, 'r') as file:
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
                if key in line:
                    tag = line.split('=')[0]
                    newline.append(tag+"= "+str(value)+end+'\n')
                    edited = True
                    if edited:
                        break
            if not edited:
                newline.append(line)

    ## make working directory and cd
    workdir = "D_"+sweep_id
    makedirs(workdir, exist_ok=True)
    chdir(workdir)

    ## make parameter file for run
    file_out = "param_"+sweep_id+".in"
    print("id",sweep_id)
    print("out",file_out)
    with open(file_out,"w") as file:
        for line in newline:
            file.writelines(line)

    ## set output file names and run fortran executable
    stdout = "stdout_"+sweep_id+".o"
    stderr = "stdout_"+sweep_id+".e"
    p = subprocess.run(["/home/links/ntt203/DCoding/DGitlab/convolutional_neural_network/bin/cnn","-f"+file_out,">"+stdout,"2>"+stderr])
    exit_code = p.wait()

    ## read from output file and log to wandb
    with open(stdout,'r') as file:
        for line in file.readlines():
            if 'epoch' in line:
                ##epoch=1, batch=1, lrate=.010, error=67.023
                result_dict = {}
                pairs = line.split(",")
                for pair in pairs:
                    key, value = pair.split("=")
                    result_dict[key.strip()] = value.strip()
                wandb.log(result_dict)

    ## return to parent directory
    chdir(curdir)


## set up sweep agents
print("Logging in...")
wandb.login()
#run = wandb.init(project='cnn_mnist_test')
print("Setting off agents")
wandb.agent(sweep_id=sweep_id, function=main, count=1, project='cnn_mnist_test')
print("All agents complete")


wandb.finish()
