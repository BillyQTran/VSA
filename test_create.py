from tqdm import tqdm
from process_large import remove_tone_line
exclude='./input/test.d'
test_file=open(exclude,'w')
corpus='/home/saplab/aivivn-vn-tones/input/demo.txt'
input_file = open(corpus, 'r')
cnt=0
for line in tqdm(input_file):
    cnt+=1
    t_line = line.strip()
    d_line = remove_tone_line(t_line)
    test_file.write(d_line+'\n')
