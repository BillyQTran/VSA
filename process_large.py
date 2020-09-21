import random

import fire
from tqdm import tqdm

from utils import remove_tone_line,normalize_tone_line,process_line


def run(corpus='/home/saplab/aivivn-vn-tones/input/corpus-full.txt',
        train_d='./input/train.d',
        train_t='./input/train.t',
        dev_d='./input/dev.d',
        dev_t='./input/dev.t',
        exclude='./input/test.d',
        dev_prob=10000.0/111000000):
    input_file = open(corpus, 'r')
    train_d_file = open(train_d, 'w')
    train_t_file = open(train_t, 'w')
    dev_d_file = open(dev_d, 'w')
    dev_t_file = open(dev_t, 'w')
    test_file=open(exclude,'w')
    ex_cnt = 0
    cnt = 0
    split=0
    for line in tqdm(input_file):
        cnt += 1
        t_line = line.strip()
        d_line = remove_tone_line(t_line)
        if random.random() < dev_prob:
            # Write to dev
            dev_d_file.write(d_line + '\n')
            dev_t_file.write(t_line + '\n')
        else:
            train_d_file.write(d_line + '\n')
            train_t_file.write(t_line + '\n')


    print('%d/%d sentences excluded' % (ex_cnt, cnt))

    input_file.close()
    train_d_file.close()
    train_t_file.close()
    dev_d_file.close()
    dev_t_file.close()


if __name__ == '__main__':
    fire.Fire(run)
