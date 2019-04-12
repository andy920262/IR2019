from arguments import get_args
from preprocess import load_all
from vsm import VSM
from query import process_query
from metrics import mean_average_precision as MAP

def build(args):
    vocab_list, doc_list, inv_file, doc_len, vocab_freq = load_all(
            args.model_dir)
    model = VSM(inv_file, doc_list, vocab_freq, doc_len)
    model.save('./model.pl')

def run(args):
    if args.build:
        model = build(args)

    if args.best:
        args.rocchio = True

    try:
        print('loading model from %s' % args.load_model)
        model = VSM(model_path=args.load_model)
    except:
        print('failed to load model, build from raw.')
        model = build(args)

    query_list, query_id = process_query(args.query_file)
    output_file = open(args.ranked_list, 'w+')
    print('query_id,retrieved_docs', file=output_file)
    for i, query in enumerate(query_list):
        doc_id, doc_score = model.get_ranking(query, args.rocchio)
        print('%s,%s' % (query_id[i], ' '.join(doc_id[:100])), file=output_file)

if __name__ == '__main__':
    args = get_args()
    run(args)
    
