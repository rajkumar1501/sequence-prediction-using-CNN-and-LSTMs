# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
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
    try:
        GeneSymbol = GeneSymbol.upper()
        seq_to_predict=[]
        
        linker = get_seq_by_GeneSymbol(GeneSymbol)
       
        
        if linker is None:
             print("no match gene found")
            
        else:
            print('No. of protein coding transcript found = '+str(len(linker)))
            count = 0
            for line in linker:
                #s = requests.session()
                count = count+1
                print('Analysing tanscript ' +str(count)+'......')
                
                id_e = line[2]
                print('Transcript ID = '+str(id_e))
                server = "https://rest.ensembl.org"
                ext = "/sequence/id/{}?".format(id_e)
                
                r = requests.get(server+ext, headers={ "Content-Type" : "text/plain","Connection": "close"})
                try:
                    if not r.ok:
                      r.raise_for_status()
                      sys.exit()
                    print('Calculating binding sites')
                    #s.keep_alive = False
                    
                    utr_seq=get_3utr(id_e)
                    #print('start_for') 
                    if utr_seq is not None:
                      
                        seq=r.text.replace('T','U')# l in negative(-l)
                        print('Length of transcript = '+str(len(seq)))
                        print("Lenght of 3'UTR = "+str(len(utr_seq))+'\n'+'\n')
                        
                        
                        if len(seq) > len(utr_seq)*1.5:
                            l=len(seq)-int(len(utr_seq)*1)
                        else:
                            l= len(seq)-len(utr_seq)
                        
                        
                        # compute minimum free energy (MFE) and corresponding structure
                        
            
                        n= RNA.pfl_fold_up(seq,8,60,120)
                        #m= RNA.pfl_fold_up(seq,16,60,120)
                        for i in range(l,len(seq)):
                            if n[i][8]>0.01: #and m[i][16]>0.001:    # n[i][4]>0.2              
                             
                                list_segment = [seq[i-23:i],seq[i-25:i],seq[i-21:i]]
                                for s in list_segment:
                                    seq_to_predict.append(s[::-1])
                    #break               
                except:                
               
                    continue
            return list(set(seq_to_predict))
    except:
        return []
                
                
            
            
    
                