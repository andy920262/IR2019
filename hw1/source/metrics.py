def mean_average_precision(retrieved, answer):
    precisions = []
    true_positives = 0
    for i in range(len(retrieved)):
        if retrieved[i] in answer:
            true_positives += 1
            precisions.append(true_positives / (i + 1))
    if len(precisions) == 0:
        return 0
    return sum(precisions) / len(answer)

if __name__ == '__main__':
    import pandas as pd
    retrieved = pd.read_csv('./ensemble.csv')[['retrieved_docs']].values
    #retrieved = pd.read_csv('./output.csv')[['retrieved_docs']].values
    answer = pd.read_csv('../queries/ans_train.csv')[['retrieved_docs']].values
    total_map = []
    for x, y in zip(retrieved, answer):
        x = x[0].strip().split(' ')
        y = y[0].strip().split(' ')
        total_map.append(mean_average_precision(x, y))
        print(total_map[-1])
    print(sum(total_map) / len(total_map))

            
        
