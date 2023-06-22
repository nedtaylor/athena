from os.path import exists
import fileinput
import wandb

file = "test.in"
if not exists(file):
    f = open(file,"x")
    # write out example file
    f.close()


def update_tag(line, key, value, end=''):
    edited = False
    if key in line:
        tag = line.split('=')[0]
        print(tag+"= "+str(value),end=end)
        print()
        edited = True
    return edited



l_dict = {
    'setup': False,
    'training': False,
    'convolution': False,
    'pool': False,
    'fullyconnected': False
}

param_dict = {
    'setup': {},
    'training': {'learning_rate': 0.5, 'momentum': 0.75, 'l1_lambda': 0.0},
    'convolution': {},
    'pool': {},
    'fullyconnected': {}
}

#for line in fileinput.FileInput(file, inplace=1):
#    if '&' in line:
#        card = line.replace('&','').strip()
#        for key in l_dict.keys():
#            if card == key:
#                l_dict[key] = True
#                break
#    elif line.strip() == '/':
#        l_dict = dict.fromkeys(l_dict,False)
#    
#    edited = False
#    true_keys = [key for key, value in l_dict.items() if value == True]
#
#    if not true_keys:
#        print(line,end='')
#        continue
#
#    if line.strip().endswith(','):
#        end = ','
#    else:
#        end = ''
#
#    for key, value in param_dict[true_keys[0]].items():
#        edited = update_tag(line, key, value,end)
#        if edited:
#            break
#    if not edited:
#        print(line,end='')


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
        

print(newline)
with open("newtest.in","w") as file:
    for line in newline:
        file.writelines(line)
