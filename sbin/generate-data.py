import os
from collections import defaultdict
import numpy as np


def print_topics(dist, vocab):
    for t, word_dist in enumerate(dist):
        print "Topic {}".format(t)
        I = np.argsort(word_dist)[::-1]
        for i in I:
            print vocab[i], word_dist[i]
        print

def print_trans(dist):
    for t, trans_dist in enumerate(dist):
        print "Topic {}".format(t)
        I = np.argsort(trans_dist)[::-1]
        for i in I:
            print i, trans_dist[i]
        print


def generate_part(num_topics, vocab, word_dist, topic_trans_dists, lam, size,
        output_path):
    def draw_word(topic):
        index = np.random.choice(len(vocab), 1, p=word_dist[topic])[0]
        return vocab[index]

    def draw_topic(topic):
        return np.random.choice(num_topics, 1, p=topic_trans_dists[topic])[0]

    sizes = np.random.poisson(lam=lam, size=size)
    
    with open(output_path, "w") as f:
        for size in sizes:

            topics = []
            words = []
            wc = defaultdict(int)
            tc = defaultdict(int)

            topic = np.random.randint(0,num_topics, 1)[0]
            word = draw_word(topic)
            topics.append(topic)
            words.append(word)
            tc[str(topic)] += 1
            wc[word] += 1

            for step in range(size-1):
                topic = np.random.randint(0,num_topics, 1)[0]
                word = draw_word(topic)
                topics.append(topic)
                words.append(word)
                tc[str(topic)] += 1
                wc[word] += 1

            word_input = " ".join(words)

            word_output = " ".join(
                sorted(wc.keys(), key=lambda x: wc[x], reverse=True))
            topic_output = " ".join(
                sorted(tc.keys(), key=lambda x: tc[x], reverse=True))
            f.write(word_input) 
            f.write(" || ")
            f.write(word_output)
            f.write(" || ")
            f.write(topic_output)
            f.write("\n")




def generate_data(num_topics, alpha, beta, lam1, lam2, 
        train_size, test_size, output_dir, seed):

    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.random.seed(seed)

    vocab = list('abcdefghijklmnopqrstuvwxyz')

    alpha = [alpha] * num_topics
    beta = [beta] * len(vocab)

    word_dist = np.random.dirichlet(beta, num_topics)
    print_topics(word_dist, vocab)
    
    topic_trans_dists = np.random.dirichlet(alpha, num_topics)
    print_trans(topic_trans_dists)

    train_path = os.path.join(output_dir, "train.toy.lam{}.txt".format(lam1))
    generate_part(
        num_topics, vocab, word_dist, topic_trans_dists, lam1, train_size,
        train_path)
    
    test_path1 = os.path.join(output_dir, "test.toy.lam{}.txt".format(lam1))
    generate_part(
        num_topics, vocab, word_dist, topic_trans_dists, lam1, test_size,
        test_path1)

    test_path2 = os.path.join(output_dir, "test.toy.lam{}.txt".format(lam2))
    generate_part(
        num_topics, vocab, word_dist, topic_trans_dists, lam2, test_size,
        test_path2)

    token_vocab_path = os.path.join(output_dir, "toy.token.vocab.txt")
    with open(token_vocab_path, "w") as f:
        f.write("\n".join(vocab))

    topic_vocab_path = os.path.join(output_dir, "toy.topic.vocab.txt")
    with open(topic_vocab_path, "w") as f:
        for i in range(num_topics):
            f.write("{}\n".format(i))

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description='Generate toy data for neural priority queue experiments')
    parser.add_argument('--topics', metavar='T', type=int,
        help='Number of topics', required=True)
    parser.add_argument('--alpha', metavar='V', type=float,
        help='Hyperparameter for topic transition matrix', required=True)
    parser.add_argument('--beta', metavar='V', type=float,
        help='Hyperparameter for distribution over words.', required=True)
    parser.add_argument('--lam1', metavar='V', type=float,
        help='Average sentence length', required=True)
    parser.add_argument('--lam2', metavar='V', type=float,
        help='Average sentence length', required=True)
    parser.add_argument('--train-size', metavar='N', type=int,
        help='Size of data set.', required=True)
    parser.add_argument('--test-size', metavar='N', type=int,
        help='Size of data set.', required=True)
    parser.add_argument('--seed', metavar='N', type=int,
        help='Random seed.', required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()
    generate_data(args.topics, args.alpha, args.beta, 
        args.lam1, args.lam2, args.train_size, args.test_size,
        args.output_dir, args.seed) 

if __name__ == "__main__":
    main()
