from transformers import BertTokenizer, BertForMaskedLM
import torch

# 选择一个预训练的BERT模型
model_name = "bert-base-uncased"  # 或者其他模型

# 加载模型和tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# 原始句子列表
input_texts = [
"a girl with long hair wearing a white dress.", #第51句
"a plate of chicken with sauce and herbs.", #第52句
"a young man sitting in front of a clock.", #第53句
       "a large jetliner flying through a cloudy blue sky.", #第54句
    "a group of people standing in front of a building.",
    "a group of waterfalls in the middle of a forest.",
"a group of ducks sitting on top of a lush green field.",
    "a group of three white rabbits sitting next to each other.",
"a woman in a red dress sitting on a ledge.",
    "a brown horse standing on a white background.",
"a man and woman crossing a street with a dog.",
    "a small rabbit is sitting in the grass.",
"a close up of a dog with a flower on its head.",
    "a black and white panda eating a leafy plant.",
"a couple of ducks standing on top of a lush green field.",
    "a couple of giraffe standing next to each other.",
"a close up of a fox in the snow.",
    "a red squirrel sitting on top of a rock.",
"an elephant with tusks walking down a dirt road.",
    "two green parrots with red heads sitting on a branch.",
"a close up of a giraffe with trees in the background.",
    "a couple of elephants standing next to a body of water.",
"a panda bear walking across a grass covered field.",
    "a red squirrel sitting on top of a tree branch.",
"a tiger walking across a lush green forest.",
    "two zebras standing next to each other in a field.",
"a white polar bear standing on top of a rocky field.",
    "a close up of a butterfly on a leaf.",
"a mouse sitting on top of a pile of books.",
    "a shark with its mouth open in the water.",
"a flock of flamingos standing on top of a lush green field.",
    "a close up of a cat with green eyes.",
"a group of giraffe standing next to a red car.",
    "a panda bear standing on top of a rock.",
"a couple of ducks floating on top of a body of water.",
    "a brown frog sitting on top of a tree branch.",
"a group of animals that are standing in the grass.",
    "a small duckling sitting on a rock.",
"a large elephant standing on top of a dry grass field.",
    "a couple of elephants walking across a dry grass field.",
"a koala bear climbing up a tree branch.",
    "a group of penguins playing in the water.",
"a group of koalas sitting on top of a lush green field.",
    "a close up of a tiger laying on the ground.",
"a brown and white dog laying on top of a blue rug.",
    "a blue truck driving down a highway with seagulls flying around.",
"a forest filled with lots of tall trees.",
    "a bald eagle flying over a body of water.",
"a close up of some white flowers on a tree.",
    "a beach with palm trees and clear blue water."
]

# 遍历每个句子
for input_text in input_texts:
    # 将句子分词
    tokens = input_text.split()

    # 以两个单词为一组遮蔽并输出预测
    for i in range(0, len(tokens)-1, 2):  # 步长为2，每次遮蔽两个单词
        masked_tokens = tokens.copy()  # 复制句子中的单词列表
        masked_tokens[i] = "[MASK]"  # 使用[MASK]替换当前位置的第一个单词
        masked_tokens[i+1] = "[MASK]"  # 使用[MASK]替换当前位置的第二个单词
        masked_text = " ".join(masked_tokens)  # 重新组合成新的句子

        inputs = tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits

        predicted_index_1 = torch.argmax(predictions[0, i + 1]).item()  # i+1是因为在tokenizer编码后会添加一个[CLS] token
        predicted_token_1 = tokenizer.convert_ids_to_tokens([predicted_index_1])[0]

        predicted_index_2 = torch.argmax(predictions[0, i + 2]).item()  # i+2是因为要预测的下一个单词
        predicted_token_2 = tokenizer.convert_ids_to_tokens([predicted_index_2])[0]

        print(f"预测遮蔽 '{tokens[i]} {tokens[i+1]}' 后的结果为: {predicted_token_1} {predicted_token_2}")
    print("\n")  # 打印完一个句子的预测结果后输出空行
