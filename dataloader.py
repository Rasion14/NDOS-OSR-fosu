from packages import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Data:
    def __init__(self,args):
        set_seed(args.seed)
        args.max_seq_length = 512

        self.data_dir = os.path.join(args.data_dir, args.dataset)
        processor = DatasetProcessor()
        self.all_labels_list = processor.get_lables(self.data_dir)
        self.num_known_cls = round(len(self.all_labels_list) * args.knowm_cls_ratio)
        self.known_labels_list = list(np.random.choice(np.array(self.all_labels_list),self.num_known_cls,replace=False))
        self.num_labels = len(self.known_labels_list)
        if args.dataset == 'oos':
            self.unseen_token = 'oos'
        else:
            self.unseen_token = '<UNK>'
        self.unseen_token_id = self.num_labels

        self.new_labels_list = self.known_labels_list + [self.unseen_token]

        self.train_examples = self.get_examples(processor,args)
        self.eval_examples = self.get_examples(processor,args,mode="eval")
        self.test_examples = self.get_examples(processor,args,mode="test")

        self.train_loader = self.get_loader(self.train_examples,args,mode="train")
        self.eval_loader = self.get_loader(self.eval_examples,args,mode="eval")
        self.test_loader = self.get_loader(self.test_examples,args,mode="test")

    def get_examples(self,processor,args,mode="train"):
        ori_examples = processor.get_examples(self.data_dir,mode)
        print(len(ori_examples))
        examples = []
        lable_map = {}
        for i,label in enumerate(self.new_labels_list):
            lable_map[label] = i
        if mode == "train":
            for example in ori_examples:
                if (example.label in self.known_labels_list) and (np.random.uniform(0, 1) <= args.labels_ratio):
                    example.label = lable_map[example.label]
                    examples.append(example)
        elif mode == "eval":
            for example in ori_examples:
                if example.label in self.known_labels_list:
                    example.label = lable_map[example.label]
                    examples.append(example)
        elif mode == "test":
            for example in ori_examples:
                if example.label in self.known_labels_list and example.label is not self.unseen_token:
                    example.label = lable_map[example.label]
                    examples.append(example)
                else:
                    example.label = self.unseen_token
                    example.label = lable_map[example.label]
                    examples.append(example)
        return examples
    def get_loader(self,examples,args,mode="train"):
        tokenizer = BertTokenizer.from_pretrained(args.pre_bert_model)
        #examples = truncation_text(examples,tokenizer,args)
        dataset = AITestDataset(examples,tokenizer,args.max_seq_length)

        if mode == "train":
            sample = RandomSampler(dataset)
            dataloader = DataLoader(dataset,batch_size=args.train_batch_size,sampler=sample)
        elif mode=="eval" or mode == "test":
            sample = SequentialSampler(dataset)
            dataloader = DataLoader(dataset,batch_size=args.eval_batch_size,sampler=sample)
        return dataloader

class Inputexample:
    def __init__(self, text_a,text_b =None,lable = None):
        self.text_a = text_a
        self.text_b = text_b
        self.label = lable


class DatasetProcessor:
    def get_examples(self, data_dir, mode):
        if mode == 'train':
            # 利用自定义方法创建一个元素是一个实例类的列表，每个类包含一个样本数据
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")  # 读取所有数据成一个列表
        elif mode == 'eval':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
    def get_lables(self,data_dir):
        total_examples = pd.read_csv(os.path.join(data_dir,"train.tsv"),sep='\t')
        labels = np.unique(np.array(total_examples["label"]))
        return labels

    def _read_tsv(self, input_file, quotechar=None):

        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(str.encode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
    def _create_examples(self, lines, set_type):

        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            if line[0] == None:
                continue
            # 使用字符串匹配式构建一个新的字符串

            text_a = line[0]
            label = line[1]

            examples.append(Inputexample(text_a,lable=label))
            random.seed(42)
            random.shuffle(examples)
            random.shuffle(examples)
            random.shuffle(examples)

        return examples


class AITestDataset(Dataset):
    def __init__(self,data,tokenizer,max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text_a = self.data[item].text_a
        label = self.data[item].label

        if self.data[item].text_b:
            encoding = self.tokenizer.encode_plus(
                text=text_a,
                text_pair=self.data[item].text_b,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )

        else:
            encoding = self.tokenizer.encode_plus(
                text=text_a,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def truncation_text(examples,tokenizer,args):
    features = []
    for example in tqdm(examples):
        if len(tokenizer.tokenize(example.text_a)) > (args.max_seq_length -2):
            total_text = tokenizer.tokenize(example.text_a)
            begin = 0
            end = args.max_seq_length - 50
            while end < len(total_text):
                texta = tokenizer.convert_tokens_to_string(total_text[begin:end])
                features.append(Inputexample(texta,text_b=example.text_b,lable=example.label))
                begin = end
                end += args.max_seq_length - 50
                if end >= len(total_text):
                    features.append(Inputexample(tokenizer.convert_tokens_to_string(total_text[begin:len(total_text)]),text_b=example.text_b,lable=example.label))
                    break
        else:
            features.append(example)
    return features




