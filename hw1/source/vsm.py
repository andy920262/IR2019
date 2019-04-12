import numpy as np
from collections import Counter
try:
    import cPickle as pickle
except ImportError:
    import pickle

class VSM(object):
    def __init__(self, inv_file=None, doc_list=None, vocab_freq=None, 
            doc_len=None, tf_type='bm25', idf_type='idf', doc_len_norm=True,
            k3=250, k1=1.2, b=0.75, model_path=None):
        if model_path != None:
            self.load(model_path)
        else:
            self.inv_file = inv_file
            self.doc_list = doc_list
            self.vocab_freq = vocab_freq
            self.doc_len = doc_len

        self.tf_type = tf_type
        self.idf_type = idf_type
        self.doc_len_norm = doc_len_norm
        self.mean_doc_len = np.mean(self.doc_len)
        self.k1 = k1
        self.b = b
        self.k3 = k3
        self._compute_idf()

    def _compute_idf(self):
        self.idf = {}
        n = len(self.doc_list)
        for term, dft in self.vocab_freq.items():
            if self.idf_type == 'no':
                self.idf[term] = 1
            elif self.idf_type == 'idf':
                self.idf[term] = np.log((n + 1) / (dft + 1)) + 1
            elif self.idf_type == 'prob_idf':
                self.idf[term] = np.log(n - dft + 1) - np.log(dft + 1) + 1
            else:
                raise ValueError('Unknown idf_type: %s' % self.idf_type)
        self.avg_idf = sum(self.idf.values()) / len(self.idf)



    def get_tf(self, term, doc_id):
        try:
            c = self.inv_file[term][doc_id]
            if self.tf_type == 'raw_tf':
                pass
            elif self.tf_type == 'bm25':
                if self.doc_len_norm:
                    pvt = self.doc_len[doc_id] / self.mean_doc_len
                    norm = 1 - self.b + self.b * pvt 
                    c = (self.k1 + 1) * c / (self.k1 * norm + c)
                else:
                    c = (self.k1 + 1) * c / (self.k1 + c) 
            else:
                raise ValueError('Unknown tf_type: %s' % self.tf_type)

            return c
        except KeyError:
            return 0

    def get_idf(self, term):
        return self.idf[term] if self.idf[term] >= 0 else 0.25 * self.avg_idf
        
    def query_vector(self, query):
        query = [term for term in query if term in self.inv_file]
        term_freq = Counter(query)
        term_list, qtf = zip(*term_freq.items())
        idf = np.array([self.get_idf(term) for term in term_list])
        qtf = np.array(qtf)
        qtf = (self.k3 + 1) * qtf / (self.k3 + qtf)
        qv = idf * qtf
        return term_list, qv

    def all_doc_vector(self, term_list):
        dvs = []
        for doc_id, doc_name in enumerate(self.doc_list):
            dv = np.array([self.get_tf(term, doc_id) for term in term_list])
            dvs.append((doc_name, dv))
        return dvs

    def get_ranking(self, query, rocchio=False, n=10, k=10):
        term_list, qv = self.query_vector(query)
        dvs = self.all_doc_vector(term_list)
        doc_rank = [(doc_id, self.score(qv, dv), dv) for doc_id, dv in dvs]
        doc_rank = sorted(doc_rank, key=lambda x: -x[1])
        if rocchio:
            rel_vec = np.mean([dv for _, _, dv in doc_rank[:k]], 0)
            #nonrel_vec = np.mean([dv for _, _, dv in doc_rank[-k:]], 0)
            term_idx = np.argsort(qv)[-n:]
            qv[term_idx] += rel_vec[term_idx]# - nonrel_vec[term_idx]
            doc_rank = [(doc_id, self.score(qv, dv), dv) for doc_id, dv in dvs]
            doc_rank = sorted(doc_rank, key=lambda x: -x[1])
        doc_id, doc_score, _ = zip(*doc_rank)
        return doc_id, doc_score

    def score(self, qv, dv):
        score = np.dot(qv, dv)
        return score #/ np.sqrt(1e-12 + np.sum(qv**2) * np.sum(dv**2))

    def save(self, path):
        pickle.dump((
                self.inv_file,
                self.doc_list,
                self.vocab_freq,
                self.doc_len
            ),
            open(path, 'wb+')
        )

    def load(self, path):
        inv_file, doc_list, vocab_freq, doc_len = pickle.load(
                open(path, 'rb'))
        self.inv_file = inv_file
        self.doc_list = doc_list
        self.vocab_freq = vocab_freq
        self.doc_len = doc_len
