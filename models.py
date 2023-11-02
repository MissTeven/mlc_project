import torch
import torch.nn as nn
from modeling_nezha import BertModel
from torch.autograd import Variable
from tokenizer import Tokenizer


class CuteBertModel(nn.Module):
    def __init__(self, pretrained_bert_path="pretrained_models/nezha-cn-base"):
        super(CuteBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, x):
        sentence = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        features, pools = self.bert(sentence, attention_mask=mask, output_all_encoded_layers=False)  # 两个都有用。
        return features


class ClassifyModel(nn.Module):
    def __init__(self, encoder=CuteBertModel(), hidden_size=768, class_num=17):
        super(ClassifyModel, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(hidden_size, class_num)

    def forward(self, x):
        return self.fc.forward(self.encoder.forward(x))


def test_cute_bert_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer(device)
    res = tokenizer.to_tensor(["快被遗忘的女演员，嫁给所有人的男神，生俩闺女日子过成这样！", "如何解读蚂蚁金服首季亏损？"])
    input_var0 = res[0].to(device)
    input_var0 = Variable(input_var0)
    input_var1 = res[1].to(device)
    input_var1 = Variable(input_var1)
    input_var2 = res[2].to(device)
    input_var2 = Variable(input_var2)
    input_var = (input_var0, input_var1, input_var2)
    print("================================input_var0==========================================")
    print(input_var0)
    print("================================embeddings==========================================")
    embeddings = BertModel.from_pretrained("pretrained_models/nezha-cn-base").embeddings
    ems = embeddings(input_var0)
    print(ems.data.shape)
    print(ems)
    print("================================input==========================================")
    print(input_var)
    model = CuteBertModel(pretrained_bert_path="pretrained_models/nezha-cn-base")
    bert_output, pooled = model.forward(input_var)
    print("================================outputs==========================================")
    print(bert_output.shape)
    print(pooled.shape)


if __name__ == '__main__':
    test_cute_bert_model()
