import os
import numpy as np
from itertools import izip
from collections import defaultdict
import random

def generate_train(num_topics, vocab_size):
    lengths = [0] * 3  + [5,6,7,8,9]
    return generate_instances(lengths, num_topics, vocab_size)

def generate_test(num_topics, vocab_size):
    lengths = [0] * 3  + [15,16,17,18,19]
    return generate_instances(lengths, num_topics, vocab_size)

def generate_instances(lengths, num_topics, vocab_size):
    np.random.shuffle(lengths)

    topics = []
    for t in xrange(num_topics):
        topic = "t{}".format(t)
        topics.extend([topic] * lengths[t])

    np.random.shuffle(topics)

    topic_output = ["t{}".format(t) for t in 
                     sorted(range(num_topics), key=lambda x: lengths[x], 
                         reverse=True)
                     if lengths[t] > 0]

    input_size = np.sum(lengths)

    words = np.random.choice(vocab_size, input_size)
    input = " ".join(["{}w{}".format(t,w) for t, w in izip(topics, words)])
    line = "{} {} {}\n".format(input, "||  ||", " ".join(topic_output))
    return line

def generate_data(output_dir, train_size, test_size, 
            seed, num_topics, vocab_size):
    
    input_vocab_path = os.path.join(output_dir, "td.input.vocab.txt")
    output_vocab_path = os.path.join(output_dir, "td.output.vocab.txt")
    train_path = os.path.join(output_dir, "td.train.txt")
    test_path = os.path.join(output_dir, "td.test.txt")

    np.random.seed(seed)
    random.seed(seed)

    with open(input_vocab_path, "w") as f:
        f.write(
                "\n".join(["t{}w{}".format(t,w)
                           for t in xrange(num_topics)
                           for w in xrange(vocab_size)]))
    with open(output_vocab_path, "w") as f:
        f.write("\n".join(["t{}".format(t) for t in xrange(num_topics)]))

    with open(train_path, "w") as f: 
        for i in range(train_size):
            train_data = generate_train(num_topics, vocab_size)
            f.write(train_data)
    with open(test_path, "w") as f: 
        for i in range(train_size):
            test_data = generate_test(num_topics, vocab_size)
            f.write(test_data)

    

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description='Generate toy data.')
    parser.add_argument('--topics', metavar='T', type=int,
        help='Number of topics', default=5)
    parser.add_argument('--vocab', metavar='V', type=int,
        help='Vocab size per topic', default=5)
    parser.add_argument('--seed', metavar='S', type=int,
        help='Random seed', default=270659512)
    parser.add_argument('--train-size', metavar='N', type=int,
        help='Size of data set.', default=100000)
    parser.add_argument('--test-size', metavar='N', type=int,
        help='Size of data set.', default=1000)
    parser.add_argument('--output-dir', type=str, required=True)
    
    args = parser.parse_args()
    if args.output_dir != "" and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)



    generate_data(args.output_dir, args.train_size, args.test_size, 
            args.seed, args.topics, args.vocab)

if __name__ == "__main__":
    main()
