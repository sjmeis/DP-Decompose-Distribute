from gensim.models import Word2Vec
import argparse
import time
from chunker import FinewebCorpusGenerator, ChunkGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-c', '--chunker')
    parser.add_argument('--chunk_workers', type=int, default=15)
    parser.add_argument('--train_workers', type=int, default=5)
    parser.add_argument('--chunksize', type=int, default=100)
    parser.add_argument('--methods', type=str, default='llr,pmi,t-score,pos,wordnet')
    args = parser.parse_args()

    if args.input == "fineweb":
        methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]

        for method in methods:
            print(f"Training {method}")

            corpus = FinewebCorpusGenerator(
                sample_size=None,
                chunker=method,
                num_processes=args.chunk_workers,
                chunksize=args.chunksize
            )

            start_time = time.perf_counter()

            model = Word2Vec(
                sentences=corpus,
                vector_size=300,
                workers=args.train_workers
            )

            save_path = f"/dev/shm/word2vec_{method}.model"
            model.save(save_path)

            end_time = time.perf_counter()
            training_time = end_time - start_time

            hours = int(training_time // 3600)
            minutes = int((training_time % 3600) // 60)
            seconds = training_time % 60

            print(f"Training completed in {hours}h {minutes}m {seconds:.2f}s")
            print(f"Vocabulary size: {len(model.wv)}")
            print(f"Saved to: {save_path}")
    else:
        chunker = ChunkGenerator(args.chunker.lower())
        sentences = chunker.load_document(args.input)
        for sent in sentences:
            print(sent)

    time.sleep(180)


if __name__ == "__main__":
    main()