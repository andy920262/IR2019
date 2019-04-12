from arguments import get_args
from preprocess import load_all
from vsm import VSM
from query import process_query
from metrics import mean_average_precision as MAP
from itertools import product

params = {
    'k1': [2.0],
    'b': [0.75],
    'k3': [500],
    'tf': ['bm25'],
    'idf': ['idf'],
    'norm': [True],
    'rocchio': [True],
    'n': [5, 10, 50],
    'k': [5, 10, 50],
}
def product_dict(d):
    return (dict(zip(d.keys(), values)) for values in product(*d.values()))

def run():
    #print([p for p in product(*params.values())])
    import pandas as pd
    model = VSM(model_path='./model.pl')
    query_list, query_id = process_query('../queries/query-train.xml')
    query_list_test, query_id_test = process_query('../queries/query-test.xml')
    answer = pd.read_csv('../queries/ans_train.csv')[['retrieved_docs']].values
    answer = [a[0].strip().split(' ') for a in answer]

    for p in product_dict(params): 
        model.k1 = p['k1']
        model.b = p['b']
        model.k3 = p['k3']
        model.tf_type = p['tf']
        model.idf_type = p['idf']
        model.doc_len_norm = p['norm']
        model._compute_idf()
        
        score = []
        for i, query in enumerate(query_list):
            doc_id, doc_score = model.get_ranking(query, p['rocchio'], p['n'], p['k'])
            score.append(MAP(doc_id[:100], answer[i]))
        score = sum(score) / len(score)
        model_str = 'k1=%.2f,b=%.2f,k3=%d,idf_type=%s,rocchio=%s,n=%d,k=%d,score=%.5f' % (p['k1'], p['b'], p['k3'], p['idf'], p['rocchio'], p['n'], p['k'], score)
        print(model_str)

        output_file = open('outputs/' + model_str + '.csv', 'w+')
        print('query_id,retrieved_docs', file=output_file)
        for i, query in enumerate(query_list_test):
            doc_id, doc_score = model.get_ranking(query, p['rocchio'], p['n'], p['k'])
            print('%s,%s' % (query_id_test[i], ' '.join(doc_id[:100])), file=output_file)

if __name__ == '__main__':
    run()
    
