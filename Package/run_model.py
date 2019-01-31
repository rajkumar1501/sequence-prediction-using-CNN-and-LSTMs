# -*- coding: utf-8 -*-
# package for running the tool
#importing Packages
import os
from mirbot_cnn import predict_mirna
from secodary_structure import get_seq_to_predict
import RNA
from Blast_predicted_mirna_seq import Blast_seq # python wapper for ViennaRNA Package. 
           #"Please look into the documentation on its website for its installation".


GeneSymbol=input("Enter Gene Symbol: ").upper()
print(GeneSymbol)
list_mrna = get_seq_to_predict(GeneSymbol)
mer_15_list = []
predict_list = set(list_mrna)
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

            
mirna_prediction=[]
pair = [] #mirna mrna pair
for mrna,mirna in list_a:
    mirna_name = Blast_seq(mirna)
    if mirna_name !=None:
        for name in mirna_name:
            mirna_prediction.append(name)
            pair.append([name,mirna,mrna])
                    
    else:
        mirna_prediction.append(mirna)
        pair.append(['Novel',mirna,mrna])
                        
                        
mirna_prediction=list(set(mirna_prediction))
predicted_pair=[]
for mir in mirna_prediction:
     for i in range(len(pair)):
         if pair[i][0]==mir:
             predicted_pair.append(pair[i])
             break
       
novel = []
known = []
for mi in mirna_prediction:
    if 'hsa'in mi:
        known.append(mi)
    else:
        novel.append(mi)


print('Total miRNA predicted: '+str(len(mirna_prediction))+'\n')
print('Total predicted mirna which are present in miRbase v22: '+str(len(known))+'\n')
print('Total Predicted novel mirna which are not present in miRbase v22: '+str(len(novel))+'\n')


if os.path.isfile(GeneSymbol+'.txt'):
    os.remove(GeneSymbol+'.txt')
with open(GeneSymbol+'.txt','w') as f:
    f.write('Gene Symbol: '+GeneSymbol+'\n'+'Predicted mirna which are present in miRbase v22:'+'\n')
    
    for name in known:
        f.write(name+'\n')

    f.write('\n\n\n'+'Predicted novel mirna which are not present in miRbase v22:'+'\n')
    for seq in novel:
        f.write(seq+'\n')
