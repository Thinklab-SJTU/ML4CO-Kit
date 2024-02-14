import os
import bz2
import lzma
import codecs
import gzip
import itertools
import numpy as np
import networkx as nx


class FileObject(object):
    def __init__(self, name, mode='r', compression=None):
        self.fp = None
        self.ctype = None
        self.fp_extra = None
        self.open(name, mode=mode, compression=compression)

    def open(self, name, mode='r', compression=None):
        if compression == 'use_ext':
            self.get_compression_type(name)
        else:
            self.ctype = compression

        if not self.ctype:
            self.fp = open(name, mode)
        elif self.ctype == 'gzip':
            self.fp = gzip.open(name, mode + 't')
        elif self.ctype == 'bzip2':
            try:
                self.fp = bz2.open(name, mode + 't')
            except:
                self.fp_extra = bz2.BZ2File(name, mode)
                if mode == 'r':
                    self.fp = codecs.getreader('ascii')(self.fp_extra)
                else:
                    self.fp = codecs.getwriter('ascii')(self.fp_extra)
        else:
            self.fp = lzma.open(name, mode=mode + 't')

    def close(self):
        if self.fp:
            self.fp.close()
            self.fp = None

        if self.fp_extra:
            self.fp_extra.close()
            self.fp_extra = None

        self.ctype = None

    def get_compression_type(self, file_name):
        ext = os.path.splitext(file_name)[1]
        if ext == '.gz':
            self.ctype = 'gzip'
        elif ext == '.bz2':
            self.ctype = 'bzip2'
        elif ext in ('.xz', '.lzma'):
            self.ctype = 'lzma'
        else:
            self.ctype = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
        
class CNF(object):
    def __init__(
        self, 
        from_file=None, 
        from_fp=None, 
        from_string=None,
        from_clauses=[], 
        from_aiger=None, 
        comment_lead=['c']
    ):
        self.nv = 0
        self.clauses = []
        self.comments = []

        if from_file:
            self.from_file(from_file, comment_lead, compressed_with='use_ext')
        elif from_fp:
            self.from_fp(from_fp, comment_lead)
        elif from_string:
            self.from_string(from_string, comment_lead)
        elif from_clauses:
            self.from_clauses(from_clauses)
        elif from_aiger:
            self.from_aiger(from_aiger)

    def __repr__(self):
        """
            State reproducible string representaion of object.
        """
        s = self.to_dimacs().replace('\n', '\\n')
        return f"CNF(from_string=\"{s}\")"

    def from_file(self, fname, comment_lead=['c'], compressed_with='use_ext'):
        with FileObject(fname, mode='r', compression=compressed_with) as fobj:
            self.from_fp(fobj.fp, comment_lead)

    def from_fp(self, file_pointer, comment_lead=['c']):
        self.nv = 0
        self.clauses = []
        self.comments = []
        comment_lead = set(['p']).union(set(comment_lead))

        for line in file_pointer:
            line = line.rstrip()
            if line:
                if line[0] not in comment_lead:
                    self.clauses.append(list(map(int, line.split()[:-1])))
                elif not line.startswith('p cnf '):
                    self.comments.append(line)

        self.nv = max(
            map(
                lambda cl: max(map(abs, cl)), 
                itertools.chain.from_iterable([[[self.nv]], self.clauses])
            )
        )
        
        
def sat_to_mis_graph(sat_path: str) -> nx.Graph:
    cnf = CNF(sat_path)
    nv = cnf.nv
    clauses = list(filter(lambda x: x, cnf.clauses))
    ind = { k:[] for k in np.concatenate([np.arange(1, nv+1), -np.arange(1, nv+1)]) }
    edges = []
    for i, clause in enumerate(clauses):
        a = clause[0]
        b = clause[1]
        c = clause[2]
        aa = 3 * i + 0
        bb = 3 * i + 1
        cc = 3 * i + 2
        ind[a].append(aa)
        ind[b].append(bb)
        ind[c].append(cc)
        edges.append((aa, bb))
        edges.append((aa, cc))
        edges.append((bb, cc))
    for i in np.arange(1, nv+1):
        for u in ind[i]:
            for v in ind[-i]:
                edges.append((u, v))
    graph = nx.from_edgelist(edges)
    return graph