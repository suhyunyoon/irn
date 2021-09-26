import os
import random

def run(args):
    random.seed(args.seed)

    with open(args.train_list, 'r') as f:
        files = f.read().split('\n')[:-1]
    
    num_data = len(files)
    
    split_data = random.sample(files, int(num_data * args.labeled_ratio))

    new_list = '\n'.join(split_data)

    new_file = os.path.join(os.path.dirname(args.train_list), 'new_' + os.path.basename(args.train_list))
    with open(new_file, 'w') as f:
        f.write(new_list)
    
    # add new list
    if args.use_unlabeled:
        args.labeled_train_list = args.train_list
        args.train_list = new_file
    else:
        args.train_list = new_file
 
