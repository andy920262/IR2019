import argparse

def get_args():
    parser = argparse.ArgumentParser(description='IR hw2') 
    parser.add_argument('-r', dest='rocchio', action='store_true')
    parser.add_argument('-b', dest='best', action='store_true')
    parser.add_argument('-i', dest='query_file', default='../queries/query-train.xml')
    parser.add_argument('-o', dest='ranked_list', default='./output.csv')
    parser.add_argument('-m', dest='model_dir', default='../model')
    parser.add_argument('-d', dest='NTCIR_dir', default='../CIRB010')
    parser.add_argument('--load_model', default='./model.pl')
    parser.add_argument('--build', action='store_true')
    return parser.parse_args()


