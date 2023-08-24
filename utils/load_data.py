import pickle


def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == '__main__':
    data = load_data(r"../dataset/en_zh.pkl")
    print(data[12666])