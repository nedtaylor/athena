from os.path import exists
from os import chdir, getcwd
import fileinput
import sys
import wandb
wandb.login()

sweep_id = '8qcw0m55'
file_template = "template.in"
curdir = getcwd() # "../"
if not exists(file_template):
    f = open(file_template,"x")
    # write out example file
    print("ERROR: "+file_template+" NOT PRESENT")
    print("Exiting...")
    f.close()
    sys.exit()


l_dict = {
    'setup': False,
    'training': False,
    'convolution': False,
    'pooling': False,
    'fullyconnected': False
}

param_dict = {
    'setup': {},
    'training': {},
    'convolution': {},
    'pooling': {},
    'fullyconnected': {}
}


def main():

    param_dict['training'] = {
        'learning_rate': sweep_config.learning_rate,
        'momentum': sweep_config.momentum,
        'l1_lambda': sweep_config.l1_lambda,
        'l2_lambda': sweep_config.l2_lambda,
        'batch_size': sweep_config.batch_size,
        'num_epochs': sweep_config.num_epochs
    }
    param_dict['convolution'] = {
        'cv_num_filters': sweep_config.cv_num_filters,
        'clip_norm': sweep_config.cv_clip_norm
    }
    param_dict['fullyconnected'] = {
        'clip_norm': sweep_config.fc_clip_norm,
        'hidden_layers': sweep_config.hidden_layers
    }

    with open(file, 'r') as file:
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
        
    file_out = "param_"+sweep_id+".in"
    with open(file_out,"w") as file:
        for line in newline:
            file.writelines(line)

    workdir = "D_"+sweep_id
    stdout = "stdout.o"
    stderr = "stdout.e"
    chdir(workdir)
    subprocess.call(["/home/links/ntt203/DCoding/DGitlab/convolutional_neural_network/bin/cnn","-f"+file_out,">"+stdout,"2>"+stderr])

    # read from output file
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

    chdir(curdir)


#wandb.init(project='cnn_mnist_test')
wandb.agent(sweep_id, function=main, count=2)
#sweep_config = wandb.config

#print(sweep_config)
#print(sweep_config.learning_rate)


wandb.finish()
