import argparse


parser=argparse.ArgumentParser(description='Arguments for RUL')


parser.add_argument('--RUL_label',type=float,default=125)
parser.add_argument('--Embedding_size',type=float,default=64)
parser.add_argument('--embedding_dim',type=float,default=64)

parser.add_argument('--timestep',type=float,default=35)
parser.add_argument('--num_factor',type=float,default=14)
parser.add_argument('--batchsize',type=float,default=256)
parser.add_argument('--dp',type=float,default=0.7)
parser.add_argument('--M',type=float,default=8)
parser.add_argument('--epoch',type=float,default=100)
parser.add_argument('--learning_rate',type=float,default=0.00001)
parser.add_argument('--weight_decay_value',type=float,default=0.00001)
parser.add_argument('--lamb',type=float,default=0.1)

configs=parser.parse_args()




