from packages import *
from dataloader import *
from bertmodel import *
from init_parameters import *

class DASMTrainModel:
    def __init__(self,args,data):
        self.model = BertEmbedModel.from_pretrained(args.pre_bert_model)
        self.classify = ClassifyLayer(args.feat_dim, data.num_labels)
        self.best_classifylayer = None
        self.best_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_eval_score = 0
        self.model.to(self.device)
        self.classify.to(self.device)


    def train(self,args,data):
        

        wait = 0
        alpha = torch.tensor(0.5)
        beta_param = torch.tensor(0.5)
        np.random.seed(args.seed)
        l = Beta(alpha,beta_param).sample((1,)).to(self.device)
        ctro_loss = nn.CrossEntropyLoss(reduction="sum")
        kl_div = torch.nn.KLDivLoss(reduction="sum")
        optimizer = torch.optim.AdamW(list(self.model.parameters())+list(self.classify.parameters()), lr=args.lr)
        num_optimizer_steps = int(len(data.train_examples) / args.train_batch_size * args.epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_optimizer_steps * args.warmup_proportion,
            num_training_steps=num_optimizer_steps
        )

        for epochs in trange(args.epochs,desc="Epoch"):
            self.model.train()
            self.classify.train()
            total_loss = 0
            num_steps = 0

            for step,batch in enumerate(tqdm(data.train_loader,desc="BoundaryTrain")):
                inputs_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                with torch.set_grad_enabled(True):
                    know_pooler_outputs,unknow_pooler_outputs,unknow_labels = self.model(
                        inputs_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels,
                        l = l,
                        num_label = data.num_labels
                    )

                    total_logits,total_logits2,total_soft_labels = self.classify(know_pooler_outputs,
                                                               unknow_pooler_outputs,
                                                               labels=labels
                                                               )
                    # print("\n",torch.sum(total_soft_labels,dim=1))
                    kl_loss = kl_div(F.log_softmax(total_logits,dim=1),total_soft_labels)
                    # close_loss = ctro_loss(total_logits,labels)
                    open_loss = ctro_loss(total_logits2,unknow_labels)
                    loss = kl_loss + 0.7 * open_loss
                    # loss = close_loss + 0.3 * open_loss
                    # loss = ctro_loss(torch.cat([total_logits,total_logits2],dim=0),
                                    #  torch.cat([labels,unknow_labels]))

                    optimizer.zero_grad()
                    nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    #释放无用变量
                    del inputs_ids, attention_mask, token_type_ids, labels
                    del know_pooler_outputs, unknow_pooler_outputs, unknow_labels
                    del total_logits, total_logits2, total_soft_labels
                    torch.cuda.empty_cache()

                    total_loss += loss.item()
                    num_steps += 1
            # print(know_pooler_outputs,"\n", unknow_pooler_outputs)
            # print('\n', total_soft_labels)
            train_loss = total_loss / num_steps
            print("train_loss:",train_loss)

            eval_score = self.eval(args,data,mode="eval")
            print("F1-score:",eval_score)

            if eval_score > self.best_eval_score:
                wait = 0
                self.best_eval_score = eval_score
                
                self.best_classifylayer = self.classify
                self.best_model = self.model


            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
        
        if self.best_model is not None:
            self.classify = self.best_classifylayer
            self.model = self.best_model
        print("best_f1:", self.best_eval_score)


    def eval(self, args, data, mode="eval"):
        
        self.classify.eval()
        self.model.eval()
        total_labels = torch.empty(0, dtype=torch.long)
        total_prebs = torch.empty(0, dtype=torch.long)
        if mode == "eval":
            dataloader = data.eval_loader
        elif mode == "test":
            dataloader = data.test_loader
        for batch in tqdm(dataloader, desc="Eval"):
            inputs_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            total_labels = torch.cat((total_labels, labels.cpu()))
            with torch.set_grad_enabled(False):
                know_pooler_outputs = self.model(
                    inputs_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,

                )
                total_logits,_ = self.classify(know_pooler_outputs)

                prebs = torch.argmax(F.log_softmax(total_logits.detach(),dim=1),dim=1)
                total_prebs = torch.cat([total_prebs,prebs.cpu()])
                
                # 释放评估阶段的中间变量
                del inputs_ids, attention_mask, token_type_ids, labels
                del know_pooler_outputs, total_logits, prebs
                torch.cuda.empty_cache()

        # print(total_logits,"\n", total_labels,"\n", total_prebs)

        y_pred = total_prebs.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        if mode == 'eval':
            cm = confusion_matrix(y_true, y_pred)
            eval_score = F_measure(cm)['F1-score']
            return eval_score

        elif mode == 'test':

            cm = confusion_matrix(y_true, y_pred)
            results = F_measure(cm)
            acc = round(accuracy_score(y_true, y_pred) * 100, 2)
            results['Accuracy'] = acc

            self.test_results = results
            print('dataset:',args.dataset)
            print('knowm_cls_ratio:',args.knowm_cls_ratio)
            print(results)
            with open(args.output_file, 'a') as f:
                f.write('dataset: ' + str(args.dataset) + '\n')
                f.write('knowm_cls_ratio: ' + str(args.knowm_cls_ratio) + '\n')
                f.write(str(results) + '\n')
                f.write('\n')



if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = init_parameters()
    args = parser.parse_args()
    # while True:
    #     try:
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024,garbage_collection_threshold:0.9"
    # max_bytes = 45*1024**3 # 你的峰值+2 GB
    # dummy = torch.empty(max_bytes//4, dtype=torch.float32, device='cuda')
    # del dummy
    # torch.cuda.empty_cache()
    datasets = ["banking","oos","stackoverflow"]
    knowm_cls_ratios = [0.25,0.50,0.75]
    for dataset in datasets:
        torch.cuda.empty_cache()
        args.dataset = dataset
        print("当前数据集：",args.dataset)
        for knowm_cls_ratio in knowm_cls_ratios:
            torch.cuda.empty_cache()
            args.knowm_cls_ratio = knowm_cls_ratio
            print("当前已知类比例：",args.knowm_cls_ratio)
            print('显卡型号（指定哪个型号时会将其设置为GPU 0）：',args.gpu_id)
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
            args.seed = 40

            for i in range(3):
                torch.cuda.empty_cache()
                # 获取各种数据
                data = Data(args)
                
                # 微调Bert模型
                print(data.num_labels)
                
                    
                print("seed:",args.seed)
                manager = DASMTrainModel(args,data)
                print('Training begin...')
                manager.train(args, data)
                print('Training finished!')

                print('Evaluation begin...')
                manager.eval(args, data, mode="test")
                print('Evaluation finished!')
                del manager
                torch.cuda.empty_cache()
                args.seed += 1
        #     break
        # except:
        #     print(f"当前显卡{args.gpu_id}显存不够，等待10秒后重新运行")
        #     print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        #     args.gpu_id = str((int(args.gpu_id)+1)%8)
        #     time.sleep(60.0)

