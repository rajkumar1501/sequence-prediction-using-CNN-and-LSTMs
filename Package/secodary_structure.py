# -*- coding: utf-8 -*-

# finding the site asseccibilty form mRNA secondery structure.
import RNA
import requests, sys
from getutr import get_3utr

import sqlite3
conn = sqlite3.connect('linker.db')
c = conn.cursor()

def get_seq_by_GeneSymbol(GeneSymbol):
    c.execute("SELECT * FROM link WHERE Gene_symbol=:Gene_symbol",{'Gene_symbol':GeneSymbol})
    return c.fetchall()

def get_seq_to_predict(GeneSymbol):
    GeneSymbol = GeneSymbol.upper()
    seq_to_predict=[]
    
    linker = get_seq_by_GeneSymbol(GeneSymbol)
   
    
    if linker is None:
         print("no match gene found")
        
    else:
        print('No. of protein coding transcript found = '+str(len(linker)))
        count = 0
        for line in linker:
         
            count = count+1
            print('Analysing tanscript ' +str(count)+'......')
            
            id_e = line[2]
            print('Transcript ID = '+str(id_e))
            server = "https://rest.ensembl.org"
            ext = "/sequence/id/{}?".format(id_e)
             
            r = requests.get(server+ext, headers={ "Content-Type" : "text/plain","Connection": "close"})
             
            if not r.ok:
              r.raise_for_status()
              sys.exit()
            print('Calculating binding sites')
            #s.keep_alive = False
            
            utr_seq=get_3utr(id_e)
            #print('start_for') 
            if utr_seq is not None:
              
                seq=r.text.replace('T','U')
                print('Length of transcript = '+str(len(seq)))
                print("Lenght of 3'UTR = "+str(len(utr_seq))+'\n'+'\n')
                
                
                if len(seq) > len(utr_seq)*1.5:
                    l=len(seq)-int(len(utr_seq)*1.5)
                else:
                    l= len(seq)-len(utr_seq)
                
                
                # compute minimum free energy (MFE) and corresponding structure
                
    
                n= RNA.pfl_fold_up(seq,16,40,80)
                for i in range(l,len(seq)):
                    if n[i][4]>0.2:                  
                     
                        s = seq[i-23:i]
                        seq_to_predict.append(s[::-1])
            else:
                continue
        return list(set(seq_to_predict))
                
                
            
            
    