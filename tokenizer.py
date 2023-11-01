import torch
# 就是全都LongTensor 转化了
from utils.official_tokenization import BertTokenizer
from torch.autograd import Variable


class Tokenizer:
    def __init__(self, pretrained_bert_path="pretrained_models/nezha-cn-base",
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_path)
        self.device = device

    def to_tensor(self, texts, pad_size=32):
        PAD, CLS = '[PAD]', '[CLS]'  # 险些坏了丞相的大事。
        pad_size = pad_size
        contents = []
        for line in texts:
            token = self.tokenizer.tokenize(line)  # 分词
            # print(len(token), token)
            token = [CLS] + token  # 句首加入CLS
            seq_len = len(token)
            mask = []
            token_ids = self.tokenizer.convert_tokens_to_ids(token)
            # print(token_ids.__len__(), token_ids)
            # 转化为数字了，这里可老重要了，需要从这里转化后才能变化成为_to_tensor
            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, int(0), seq_len, mask))

        # print(contents)
        x = torch.LongTensor([_[0] for _ in contents]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in contents]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in contents]).to(self.device)
        return (x, seq_len, mask)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer()
    res = tokenizer.to_tensor(["快被遗忘的女演员，嫁给所有人的男神，生俩闺女日子过成这样！"])
    print("-----res----")
    print(res)
    print("-----input_var0----")
    input_var0 = res[0].to(device)
    print(input_var0)
    input_var0 = Variable(input_var0)
    print(input_var0)
    print("-----input_var1----")
    input_var1 = res[1].to(device)
    print(input_var1)
    input_var1 = Variable(input_var1)
    print(input_var1)
    print("-----input_var2----")
    input_var2 = res[2].to(device)
    print(input_var2)
    input_var2 = Variable(input_var2)
    print(input_var2)

    print("-----input_var----")
    input_var = (input_var0, input_var1, input_var2)
    print(input_var)
