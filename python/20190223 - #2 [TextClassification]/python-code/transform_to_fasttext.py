#https://hackernoon.com/text-classification-simplified-with-facebooks-fasttext-b9d3022ac9cb
#https://fasttext.cc/docs/en/supervised-tutorial.html
import csv
from fastText import train_supervised, load_model



def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


def parse_train_to_ft_format(input_file, output_train, output_test):
    with open(input_file) as f:
        file = csv.reader(f, delimiter=',', quotechar='"')
        lines = [line for line in file][1:]
        lines = ['__label__%s %s' % (line[2], line[1]) for line in lines]
        split = int(0.8 * len(lines))
        lines_test = lines[split:]
        lines_train = lines[:split]
        with open(output_train, 'w') as f2:
            for line in lines_train:
                f2.write(line + "\n")

        with open(output_test, 'w') as f2:
            for line in lines_test:
                f2.write(line + "\n")


if __name__ == '__main__':
    train_data = 'data/quora/fast_text.train'
    valid_data = 'data/quora/fast_text.test'
    parse_train_to_ft_format('data/quora/train.csv', train_data, valid_data)

    # train_supervised uses the same arguments and defaults as the fastText cli
    model = train_supervised(
        input=train_data, epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1
    )
    print_results(*model.test(valid_data))

    model = train_supervised(
        input=train_data, epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1,
        loss="hs"
    )
    print_results(*model.test(valid_data))

    model.save_model("models/quora.bin")
    model.quantize(input=train_data, qnorm=True, retrain=True, cutoff=100000)
    print_results(*model.test(valid_data))
    model.save_model("models/quora.ftz")

    print(model.test_label(valid_data))
    print(model.test('data/quora/fast_text.test'))

    # BIN
    # {'__label__1': {'precision': 0.6188333727997474, 'recall': 0.48186846957590657, 'f1score': 0.5418293652164898},
    # '__label__0': {'precision': 0.9660841017718341, 'recall': 0.9802861750117369, 'f1score': 0.9731333242825387}}