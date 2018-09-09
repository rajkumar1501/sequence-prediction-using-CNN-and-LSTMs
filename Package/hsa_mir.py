# -*- coding: utf-8 -*-

import sqlite3
conn = sqlite3.connect('hsa_mir.db')


    

c = conn.cursor()


#    

def insert_seq(Id,seq):
    
        
    c.execute("INSERT INTO hsa_mir_seq VALUES (:Mir_ID , :seq )", {'Mir_ID': Id, 
                      'seq': seq})
   

def get_seq_by_mir_id(Id):
    
    c.execute("SELECT * FROM hsa_mir_seq  WHERE Mir_ID=:Mir_ID",{'Mir_ID':Id})
    return c.fetchall()    
    
def get_all_data():
    c = conn.cursor()
    c.execute("SELECT * FROM hsa_mir_seq")
    return c.fetchall()  

#
#for seq_record in SeqIO.parse("mature.fa", "fasta"):
#    Id = seq_record.description.split()[0]
#
#    #print(Id,str(seq_record.seq))
#    
#    if 'hsa' in Id: 
#        insert_seq(Id,str(seq_record.seq))
#        print(Id,str(seq_record.seq))
#conn.commit()    

   
   




#seq_object=get_seq_by_mir_id('hsa-miR-548b-5p')
#print(len(seq_object))
c.close()
# -*- coding: utf-8 -*-

