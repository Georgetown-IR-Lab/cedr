import os
import sys
import argparse
from tqdm import tqdm

def indri_doc_extractor(path):
    import pyndri
    index = pyndri.Index(path)
    id2token = index.get_dictionary()[1]
    def wrapped(docid):
        doc_id_tuples = index.document_ids([docid])
        if not doc_id_tuples:
            return None # not found
        int_docid = doc_id_tuples[0][1]
        _, doc_toks = index.document(int_docid)
        return ' '.join(id2token[tok] for tok in doc_toks if tok != 0)
    return wrapped


def lucene_doc_extractor(path):
    import jnius_config
    if not os.path.exists('bin/anserini.jar'):
        sys.stderr.write('missing bin/anserini.jar')
        sys.exit(1)
    jnius_config.set_classpath("bin/anserini.jar")
    from jnius import autoclass
    index_utils = autoclass('io.anserini.index.IndexUtils')(path)
    def wrapped(docid):
        lucene_doc_id = index_utils.convertDocidToLuceneDocid(docid)
        if lucene_doc_id == -1:
            return None # not found
        return index_utils.getTransformedDocument(docid)
    return wrapped


INDEX_MAP = {
    'indri': indri_doc_extractor,
    'lucene': lucene_doc_extractor
}


def main_cli():
    parser = argparse.ArgumentParser('Extract documents from index (stdin: document IDs, '
                                     'stdout: datafile, stderr: progress and missing documents)')
    parser.add_argument('index_type', choices=INDEX_MAP.keys())
    parser.add_argument('index_path')
    args = parser.parse_args()
    doc_extractor = INDEX_MAP[args.index_type](args.index_path)
    for docid in tqdm(sys.stdin):
        docid = docid.rstrip()
        doc = doc_extractor(docid)
        if doc is None:
            tqdm.write(f'[WARN] missing doc id: {docid}')
        else:
            doc = doc.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
            sys.stdout.write(f'doc\t{docid}\t{doc}\n')


if __name__ == '__main__':
    main_cli()
