import os

def load_vocab(path):
    print('loading %s' % path)
    file = open(path, 'r')
    vocab = file.read().splitlines()
    file.close()
    return vocab

def load_file_list(path):
    print('loading %s' % path)
    file = open(path, 'r')
    file_list = file.read().splitlines()
    file_list = [line.split('/')[-1].lower() for line in file_list]
    file.close()
    return file_list

def load_inv_file(path, vocab, file_list):
    print('loading %s' % path)
    file = open(path, 'r')
    lines = file.read().splitlines()
    file.close()
    
    inv_file = {}
    doc_len = [0 for _ in range(len(file_list))]
    voc_freq = {v : 0 for v in vocab}
    for line in lines:
        line = line.strip().split(' ')
        if len(line) == 3:
            gram = {}
            if line[1] == '-1': # unigram
                key = vocab[int(line[0])]
            else: # bigram
                key = vocab[int(line[0])] + vocab[int(line[1])]
            inv_file[key] = gram
        else:
            doc_id, count = int(line[0]), int(line[1])
            gram[doc_id] = count
            doc_len[doc_id] += count
            if key not in voc_freq:
                voc_freq[key] = 0
            voc_freq[key] += 1


    return inv_file, doc_len, voc_freq

def load_all(model_dir):
    vocab = load_vocab(os.path.join(model_dir, 'vocab.all'))
    file_list = load_file_list(os.path.join(model_dir, 'file-list'))
    inv_file, doc_len, voc_freq = load_inv_file(os.path.join(model_dir, 'inverted-file'), vocab, file_list)
    return vocab, file_list, inv_file, doc_len, voc_freq

