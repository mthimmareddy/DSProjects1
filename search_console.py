
	

import os
import re
import fnmatch
from datetime import datetime
path='C:\\Users\\rishi\\Desktop\\Search_Engine'
Programs = ''' 
    First argument:
        
    1.Search based on filename
    2.Search based on file content
    3.Search based on file creation/modified date
    4.Search based on file size
    5.Search based on file type(pdf,img,doc,txt)
    
    c= 
    1.whole filename
    2.begining of filename
    3.end of filename
    4.anywhere in filename
    
   
    
    python search.py ('Enter the Search criteria :') ('Enter the pattern based on Search criteria :')
    Example:    python search.py 1 "filename","c"
                python search.py 2 "keyword in file" "filename"
                python search.py 3 "creation/modification" "date of creation"
                python search.py 4 "size of file"
                python search.py 5 "type of file(doc/pdf/img/)"
  
    '''
    

count=[]



def searchbasedonfilename(pattern,c):
    flag=0
    
    #c=int(input('Where to find the pattern:'))
    for root,dirs,files in os.walk(path):
        for name in files:
            #print(os.path.join(root, name))
            if c==1 and fnmatch.fnmatch(name,pattern):
                
                flag=1
                count.append(os.path.join(root, name))
            elif c==2 and name.startswith(pattern):
                flag=1
                count.append(os.path.join(root, name))
                
            elif c==3 and name.endswith(pattern):
                flag=1
                count.append(os.path.join(root, name))
                
            elif c==4 and pattern in name:
                flag=1
                count.append(os.path.join(root, name))
                

    if(flag):
        #print('Fallowing files found in path :{0}'.format(count))
        return count
    else:
        print('File not found')
        return None
       
def pdfreadder(file):
    import PyPDF2
    pdf_file = open(file, 'rb')
    read_pdf = PyPDF2.PdfFileReader(pdf_file)
    number_of_pages = read_pdf.getNumPages()
    page = read_pdf.getPage(0)
    page_content = page.extractText()
    print (page_content.encode('utf-8'))
    return page_content


       

def searchbasedonfilecontent(pattern,filename):
    #file=input('Enter the file in which pattern to be searched') 
    count=searchbasedonfilename(filename,4)
   
    
    flag1=0
    print(count)
    if count:
        for file in count:
          if 'pdf' in file:
            content=pdfreadder(file)
            content=content.split(";")
            with open('tem.txt','w') as f1:
             for ele in content:
              f1.write(ele)
              f1.write('\n')
            with open('tem.txt', 'r') as f2:
             for line in f2.readlines():
              #rint(line)
              match=re.search(pattern,line)
              #rint(match)
              if(match):
                 flag1=1
                 print(line)
                 with open("temp.txt",'w') as f:
                     f.write(line)
                     
          else:
           for file in count:
               with open(file, 'r') as f2:
                   for line in f2.readlines():
                       match=re.search(pattern,line)
                       if(match):
                           flag1=1
                           print (line)
                           with open("temp.txt",'w') as f:
                               f.write(line)
               
          if(flag1):
             print('Pattern found')
             return "temp.txt"
          else:
             print('Pattern Not found')
             return None
          
                 
    else: 
        print('File not found')
        return None

def searchbasedondate(pattern):
    count=[]
    #path='C:\\Users\\rishi\\Desktop\\Search_Engine'
    for root,dirs,files in os.walk(path):
        for directory in dirs:
            ctime=datetime.fromtimestamp(os.path.getctime(directory)).strftime('%d-%m-%Y')
            if (ctime==pattern):
                print("Folder is created on date {0}".format(directory))
           
        for name in files:
            path1=os.path.join(root,name)
            #ctime=datetime.fromtimestamp(os.path.getctime(path1)).strftime('%d-%m-%Y')
            #print(name,ctime)
            if (ctime==pattern):
                print("Name of file created on date {0}".format(path1))
                count.append(path1)
    return count
           
             
                
def searchbasedonsize(pattern):
    count=[]
    
    for root,dirs,files in os.walk(path):
        for directory in dirs:
            size=os.path.getsize(directory)
            if (size==pattern):
                print("Folder is created on date {0}".format(directory))
           
        for name in files:
            path1=os.path.join(root,name)
            size=os.path.getsize(path1)
            if (size==pattern):
                print("Folder is created on date {0}".format(path1))
                count.append(path1)
    return count
			
def searchbasedontypeoffile(pattern):
    count_files=[]
    for root,dirs,files in os.walk(path):
        c1=0
        c2=0
        c3=0
        c4=0
        count=[]
        
        for f1 in files:
            path1=os.path.join(root,f1)
            if f1.endswith(".txt") and (pattern in f1):
                c1=c1+1
                count_files.append(path1)
            if (f1.endswith(".img") or f1.endswith(".png") or f1.endswith(".jpeg") or f1.endswith(".mp3"))and (pattern in f1):
                c2=c2+1
                count_files.append(path1)
            if f1.endswith(".pdf") and (pattern in f1):
                c3=c3+1
                count_files.append(path1)
            if f1.endswith(".doc") and (pattern in f1):
                c4=c4+1
                count_files.append(path1)
       
        count.append(c1)
        count.append(c2)
        count.append(c3)
        count.append(c4)
       
    return count_files 


import sys
if __name__=='__main__':
    
    print(Programs)
    
     
    
    if int(sys.argv[1])==1:
        result=searchbasedonfilename(sys.argv[2],int(sys.argv[3]))
    elif int(sys.argv[1])==2:
        result=searchbasedonfilecontent(sys.argv[2],sys.argv[3])
    elif int(sys.argv[1])==3:
        result=searchbasedondate(sys.argv[2])
    elif int(sys.argv[1])==4:
        result=searchbasedonsize(sys.argv[2])
    elif int(sys.argv[1])==5:
        result=searchbasedontypeoffile(sys.argv[2])
    else:
        print("Search criteria range from 1-5")
        
        
    print("\n*****************************************************************************")
    
    print("\nThe output of search result is:{0}".format(result))
    print("\n*****************************************************************************\n")
        
  





 

