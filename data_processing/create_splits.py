from util.misc import read_list, write_list
import random


def create_split():
    f_train, f_val, f_train_val = 0.75, 0.25, 0.20
    n_vis_val, n_vis_train = 35, 15
    folder = 'data/splits/3D-FUTURE/Sofa/'
    all_samples = read_list(folder + 'all.txt')
    error = set(read_list(folder + 'error.txt'))
    all_samples = [x for x in all_samples if x not in error]
    train = random.sample(all_samples, int(len(all_samples) * f_train))
    val = random.sample([x for x in all_samples if x not in train], int(len(all_samples) * f_val))
    remaining = [x for x in all_samples if x not in train and x not in val]
    train.extend(remaining)
    train_val = random.sample(train, int(len(train) * f_train_val))
    vis_train = random.sample(val, n_vis_train)
    vis_val = random.sample(train, n_vis_val)
    print(f"({len(train)}, {len(val)}, {len(train_val)}, {len(vis_train)}, {len(vis_val)})")
    write_list(folder + 'train.txt', train)
    write_list(folder + 'val.txt', val)
    write_list(folder + 'train_val.txt', train_val)
    write_list(folder + 'vis_val.txt', vis_val)
    write_list(folder + 'vis_train.txt', vis_train)


if __name__ == "__main__":
    create_split()
