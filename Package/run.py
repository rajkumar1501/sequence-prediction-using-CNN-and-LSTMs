# -*- coding: utf-8 -*-
# package for running the tool
#importing Packages
import Levenshtein
from mirbot_cnn import predict_mirna
from secodary_structure import get_seq_to_predict
from hsa_mir import get_all_data
import RNA # python wapper for ViennaRNA Package. 
           #"Please look into the documentation on its website for its installation".


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
                        
                        
                
                
                
print('Predicting miRNA sequences.......\n')             
for mrna in predict_list:
    micro= predict_mirna(mrna)[:-1]
    
    mrna=mrna[::-1]
    if RNA.fold(mrna[:]+'LLLLLLLL'+micro[:])[1]<-7:
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
    counter  = 0
    for seq in mirna_seq_list:
        if Levenshtein.ratio(predict_list[i][1][:15], seq[1][:15])>0.795:
            
                        
                        
                            micro=seq[1]
                            mrna=predict_list[i][0][::-1]
                            if RNA.fold(mrna[:]+'LLLLLLLL'+micro[:])[1]<-5:
                                predicted_mirna.append(micro)
                                mirna_prediction.append(seq[0])
                                pair.append([seq[0],seq[1],predict_list[i][0]])
                                counter=counter+1
        
                    
        if mirna_seq==False & counter==0:
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

print('Total miRNA predicted: '+str(len(mirna_prediction))+'\n')
print('Total predicted mirna which are present in miRbase v22: '+str(len(known))+'\n')
print('Total predicted novel mirna which are not present in miRbase v22: '+str(len(novel))+'\n')
# -*- coding: utf-8 -*-

