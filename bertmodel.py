from packages import *
from dataloader import *



class BertEmbedModel(BertPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        # self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.config = config

        self.dense1 = torch.nn.Linear(config.hidden_size,768)
        # self.dense2 = torch.nn.Linear(512,512)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.activation = torch.nn.ReLU()
        # self.classifier = torch.nn.Linear(512,config.num_labels)
        # self.centers = nn.Parameter(torch.randn(data.num_labels,256))
        self.post_init()
    def forward(self,
                inputs_ids,
                token_type_ids=None,
                attention_mask=None,
                labels = None,
                l = None,
                num_label = None
                ):

        outputs = self.bert(
            inputs_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        pooler_outputs = torch.mean(outputs.last_hidden_state,dim=1)




        # 正样本数据编码
        # pooler_outputs = self.dense1(pooler_outputs)
        # pooler_outputs = self.activation(pooler_outputs)
        # pooler_outputs = self.dropout(pooler_outputs)
        
        # L2 = F.normalize(pooler_outputs,2,dim=1)
        # know_pooler_outputs = L2


        know_pooler_outputs = pooler_outputs


        if labels is not None:
            unknow_pooler_outputs = []
            for i,label in enumerate(labels):
                indices = torch.where(labels != label)[0]
                # distances = torch.cdist(pooler_outputs[i],pooler_outputs[indices])
                # min_indices = indices[torch.argmin(distances,dim=1)]

                #选择距离最小的未知类样本
                un_pooler = l * pooler_outputs[i] + (1-l) * pooler_outputs[indices]
                distances = torch.norm(pooler_outputs[i]-un_pooler,p=2,dim=1)
                un_pooler = un_pooler[torch.argmin(distances)]
                
                #随机选择
                # un_pooler = l * pooler_outputs[i] + (1-l) * pooler_outputs[indices]
                # row_index = torch.randint(len(un_pooler),(1,)).item()
                # un_pooler = un_pooler[row_index]

                #选取最大样本
                # un_pooler = l * pooler_outputs[i] + (1-l) * pooler_outputs[indices]
                # distances = torch.norm(pooler_outputs[i]-un_pooler,p=2,dim=1)
                # un_pooler = un_pooler[torch.argmax(distances)]

                unknow_pooler_outputs.append(un_pooler)
            unknow_pooler_outputs = torch.stack(unknow_pooler_outputs)
            unknow_labels = torch.tensor([num_label]*len(labels),dtype=torch.long).to(labels.device)
            # total_labels = torch.cat([labels,unknow_labels])

            # num = int( len(labels) / 2 - 5)
            # new_indices = torch.randint(low=0,high=len(labels)-1,size=(num,))
            return know_pooler_outputs,unknow_pooler_outputs,unknow_labels
        else:
            return know_pooler_outputs


class ClassifyLayer(nn.Module):
    def __init__(self,inputs_dim,num_class,phi=0.3):
        super(ClassifyLayer, self).__init__()
        self.num_labels = num_class
        # self.know_classifylayer = nn.Linear(inputs_dim,num_labels)
        # self.unknow_classifylayer = nn.Linear(inputs_dim,1)
        self.classifylayer = nn.Linear(inputs_dim,num_class+1)
        # self.unknow_classifylayer = nn.Linear(inputs_dim,num_labels + 1)

        self.phi = phi



    def forward(self,know_feature,unknow_feature=None,labels=None):
        # know_logits = self.know_classifylayer(know_feature)
        # unknow_logits = self.unknow_classifylayer(know_feature)
        # total_logits = torch.cat([know_logits,unknow_logits],dim=1)
        total_logits = self.classifylayer(know_feature)
        if unknow_feature is not None:
            # know_logits2 = self.know_classifylayer(unknow_feature)
            # unknow_logits2 = self.unknow_classifylayer(unknow_feature)
            # total_logits2 = torch.cat([know_logits2,unknow_logits2],dim=1)
            total_logits2 = self.classifylayer(unknow_feature)
        else:
            total_logits2 = None
        # unknow_logits = self.unknow_classifylayer(unknow_feature)
        # know_logits = F.log_softmax(know_logits,dim=1)
        # unknow_logits = F.log_softmax(unknow_logits,dim=1)
        # total_logits = torch.cat([know_logits,unknow_logits],dim=1)
        if labels is not None:

            soft_know_labels = torch.zeros((len(labels),self.num_labels),device=know_feature.device)

            soft_know_labels[torch.arange(len(labels),device=know_feature.device), labels] = 1 - self.phi


            soft_unknow_labels = torch.full((len(labels),), self.phi, device=know_feature.device).unsqueeze(1)

            total_soft_labels = torch.cat([soft_know_labels,soft_unknow_labels],dim=1).float()

            return total_logits,total_logits2,total_soft_labels
        else:
            return total_logits,total_logits2



