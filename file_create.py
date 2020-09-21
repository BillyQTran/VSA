from process_large import run
run(corpus='/home/saplab/aivivn-vn-tones/input/corpus-full.txt',
        train_d='./input/train.d',
        train_t='./input/train.t',
        dev_d='./input/dev.d',
        dev_t='./input/dev.t',
        exclude='./input/test.d',
        dev_prob=10000.0/111000000)
