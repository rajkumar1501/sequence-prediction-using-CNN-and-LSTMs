# -*- coding: utf-8 -*-


#from difflib import SequenceMatcher
import Levenshtein
from mirbot_cnn import predict_mirna
from secodary_structure import get_seq_to_predict
from hsa_mir import get_all_data
import RNA
#from joblib import Parallel, delayed
#file = open('gene.txt','r')
##file = open('mini.txt','r')
#list_of_gene = file.readlines()
#list_of_gene = [i.strip() for i in list_of_gene]
#result_file = open('result.txt','w+')
#for GeneSymbol in list_of_gene:
#    

GeneSymbol=input("Enter Gene Symbol: ").upper()
list_mrna = get_seq_to_predict(GeneSymbol)
mer_15_list = []
predict_list = []
for mrna in list_mrna:
    if mrna[:15] not in mer_15_list:
        mer_15_list.append(mrna[:20])
        predict_list.append(mrna)
predicted_mirna=[]
print(str(len(predict_list))+" mRNA segments were found\n")
                        
                        
                
                
                
print('Predicting miRNA sequences.......\n')              #Parallel(n_jobs=4)(delayed(predicted_mirna.append)(predict_mirna(mrna)) for mrna in list_mrna)
for mrna in predict_list:
    micro= predict_mirna(mrna)[:-1]
    
    mrna=mrna[::-1]
    if RNA.fold(mrna[:]+'LLLLLLLL'+micro[:])[1]<-3:
        predicted_mirna.append(micro)

list_a=[]
for mrna,mirna in zip(predict_list,predicted_mirna):
    list_a.append([mrna,mirna])
                    
predicted_mirna=list(set(predicted_mirna))
mer_15_list = []
predict_list = []
for i in range(len(list_a)):
    if list_a[i][1][:15] not in mer_15_list:
                        mer_15_list.append(list_a[i][1][:15])
                        predict_list.append(list_a[i])
mirna_seq_list = get_all_data()
list_a=0
mirna_prediction=[]
pair = []
for i in range(len(predict_list)):
    mirna_seq=False
    #counter  = 0
    for seq in mirna_seq_list:
        if Levenshtein.ratio(predict_list[i][1][:15], seq[1][:15])>0.795:
                        
                        
                            micro=seq[1]
                            mrna=predict_list[i][0][::-1]
                            #pair.append([seq[0],predict_list[i][0]])
                            if RNA.fold(mrna[:]+'LLLLLLLL'+micro[:])[1]<0:
                                predicted_mirna.append(micro)
                                mirna_prediction.append(seq[0])
                                pair.append([seq[0],seq[1],predict_list[i][0]])
                                #counter=counter+1
                            
        #else:
            #mirna_seq=False
                    
        if mirna_seq==False:
                        mirna_prediction.append(predict_list[i][1])
                        pair.append([predict_list[i][1],predict_list[i][0]])
                        
                        
mirna_prediction=list(set(mirna_prediction))
predicted_pair=[]
for mir in mirna_prediction:
     for i in range(len(pair)):
         if pair[i][0]==mir:
             predicted_pair.append(pair[i])
             break
pair=0         
novel = []
known = []
for mi in mirna_prediction:
    if 'hsa'in mi:
        known.append(mi)
    else:
        novel.append(mi)
with open('gene_to_mirna.csv','r') as f:
    lines = f.read().split('\n')
                
gene_to_mi_dic = {}
for line in lines:
                    gene,mir=line.split(',')
                    if gene not in gene_to_mi_dic.keys():
                        gene_to_mi_dic[gene]=[mir]
                    else:
                        gene_to_mi_dic[gene].append(mir)
                    
l=list(set(mirna_prediction).intersection(gene_to_mi_dic[GeneSymbol]))
percentage = (len(l)/len(gene_to_mi_dic[GeneSymbol]))*100
print(str(len(l))+','+str(percentage)+','+str(len(gene_to_mi_dic[GeneSymbol]))+','+str(l)+'\n')
print(GeneSymbol)
print(percentage)
# -*- coding: utf-8 -*-

