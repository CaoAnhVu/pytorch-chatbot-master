import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
if __name__ == '__main__':
    with open('D:\TRÍ TUỆ NHÂN TẠO\pytorch-chatbot-master\pytorch-chatbot-master\intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []
    # Lặp qua từng câu trong các mẫu ý định của mình
    for intent in intents['intents']:
        tag = intent['tag']
        # Thêm vào danh sách thẻ
        tags.append(tag)
        for pattern in intent['patterns']:
            # Mã hóa từng từ trong câu
            w = tokenize(pattern)
            # Thêm vào danh sách từ của chúng 
            all_words.extend(w)
            # Thêm vào cặp XY
            xy.append((w, tag))

    # Thân và hạ thấp từng từ
    ignore_words = ['?', '.', '!',]
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    # Loại bỏ trùng lặp và sắp xếp
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    print(len(xy), "patterns")
    print(len(tags), "tags:", tags)
    print(len(all_words), "unique stemmed words:", all_words)

    # Tạo dữ liệu đào tạo
    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        # X: túi từ cho mỗi pattern_sentence
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        # y: PyTorch CrossEntropyLoss chỉ cần nhãn lớp, không phải một hot
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Siêu tham số, truyền vào tham số
    num_epochs = 1000
    batch_size = 8
    learning_rate = 0.001
    input_size = len(X_train[0])
    hidden_size = 8
    output_size = len(tags)
    print(input_size, output_size)


    class ChatDataset(Dataset):

        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = y_train

        # Hỗ trợ lập chỉ mục sao cho tập dữ liệu [i] có thể được sử dụng để lấy mẫu thứ i
        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        # Gọi len(dataset) để trả về kích thước
        def __len__(self):
            return self.n_samples

    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True
                            )
    #num_workers=2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Mất mát và tối ưu hóa
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Đào tạo mô hình
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            
            # Chuyền về phía trước
            outputs = model(words)
            # Nếu y là một hot, chúng ta phải áp dụng
            # labels = torch.max(labels, 1)[1]
            loss = criterion(outputs, labels)
            
            # Lạc hậu và tối ưu hóa
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


    print(f'final loss: {loss.item():.4f}')

    data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
    }


    FILE = "data.pth"
    torch.save(data, FILE)

    print(f'training complete. file saved to {FILE}')
