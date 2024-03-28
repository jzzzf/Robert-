# classification
一个分类任务,目前做的是糖尿病的问题分类,baseline模型是chinese-roberta-wwm-ext,f1得分:0.6922562941397946
主要做了几个修改,一是模型融合,把3个模型的结果投票来决定最终的预测结果,二是采用了focal loss,提升后f1得分:0.7381336411842875
