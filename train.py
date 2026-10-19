"""Train 100-dimensional skip-gram embeddings on the complete text8 corpus."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import time
import urllib.request
import zipfile

import gensim
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus

ROOT = Path(__file__).resolve().parent
URL = 'https://mattmahoney.net/dc/text8.zip'
MD5 = '3bea1919949baf155f99411df5fada7e'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.workers < 1:
        parser.error('epochs and workers must be positive')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    data = ROOT / 'data'
    output = ROOT / 'artifacts'
    data.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    corpus = data / 'text8'
    if not corpus.exists():
        archive = data / 'text8.zip'
        if not archive.exists():
            temporary = archive.with_suffix('.zip.part')
            urllib.request.urlretrieve(URL, temporary)
            temporary.replace(archive)
        with zipfile.ZipFile(archive) as z:
            with z.open('text8') as source, corpus.open('wb') as target:
                import shutil
                shutil.copyfileobj(source, target)
    digest = hashlib.md5(corpus.read_bytes()).hexdigest()
    if digest != MD5 or corpus.stat().st_size != 100_000_000:
        raise ValueError('text8 size/checksum mismatch; remove data/text8 and retry')
    config = dict(vector_size=100, window=5, min_count=5, sg=1, negative=5,
                  hs=0, sample=1e-3, alpha=0.025, min_alpha=0.0001,
                  epochs=args.epochs, workers=args.workers, seed=args.seed)
    started = time.time()
    model = Word2Vec(sentences=Text8Corpus(str(corpus)), **config)
    model.save(str(output / 'word2vec.model'))
    model.wv.save(str(output / 'word2vec.kv'))
    model.wv.save_word2vec_format(str(output / 'text8-100d.bin'), binary=True)
    model.wv.save_word2vec_format(str(output / 'text8-100d.txt'), binary=False)
    # Reload the portable binary to verify export, dimensions and finite values.
    vectors = gensim.models.KeyedVectors.load_word2vec_format(
        str(output / 'text8-100d.bin'), binary=True)
    assert vectors.vectors.shape == (len(model.wv), 100)
    assert np.isfinite(vectors.vectors).all()
    assert np.all(np.linalg.norm(vectors.vectors, axis=1) > 0)
    assert np.allclose(vectors['king'], model.wv['king'])
    report = dict(dataset_url=URL, dataset_md5=digest, dataset_bytes=corpus.stat().st_size,
                  corpus_tokens=model.corpus_total_words, vocabulary_size=len(vectors),
                  config=config, training_seconds=round(time.time() - started, 2),
                  gensim_version=gensim.__version__, python_version=platform.python_version(),
                  shape=list(vectors.vectors.shape), all_finite=True,
                  neighbors={word: vectors.most_similar(word, topn=5)
                             for word in ['king', 'computer', 'music']})
    (output / 'training_report.json').write_text(json.dumps(report, indent=2) + '\n')
    (ROOT / 'training_report.json').write_text(json.dumps(report, indent=2) + '\n')
    logging.info('Completed: %s', report)


if __name__ == '__main__':
    main()
