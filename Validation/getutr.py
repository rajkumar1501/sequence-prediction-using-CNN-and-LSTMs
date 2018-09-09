
import requests


def get_3utr(transcript_id):
    link = ('https://asia.ensembl.org/Homo_sapiens/Export/Output/Transcript?db=core;'+
    'flank3_display=0;flank5_display=0;output=fasta;strand=feature;'+
    't={}'.format(transcript_id)+';param=utr3;genomic=unmasked;_format=Text')
    utr = requests.get(link)
    
    
       
    utr = utr.text.split('>')[1]
    utr_split = utr.split('\n')
    utr_seq=''
    #print('start_for') 
    if 'utr3'in utr_split[0]:
        for fasta in utr_split[1:]:
            utr_seq=utr_seq+fasta.replace('\n','')
            
        return(utr_seq)    
    else:
        return(None)

# -*- coding: utf-8 -*-

