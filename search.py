
	

import os
import re
import fnmatch
from datetime import datetime
path='C:\\Users\\rishi\\Desktop\\Search_Engine'
Programs = '''
1.Search based on filename
2.Search based on file content
3.Search based on file creation/modified date
4.Search based on file size
5.Search based on file type(pdf,img,doc,txt)

'''

count=[]



def searchbasedonfilename(pattern):
    flag=0
    x= '''
    1.whole filename
    2.begining of filename
    3.end of filename
    4.anywhere in filename
    '''
    print (x)
    c=int(input('Where to find the pattern:'))
    for root,dirs,files in os.walk(path):
        for name in files:
            print(os.path.join(root, name))
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
                

        for name in dirs:
            print(os.path.join(root, name))
    if(flag):
        print('Fallowing files found in path :{0}'.format(count))
    else:
        print('File not found')
    return flag,count[0]
       
def pdfreadder(file):
    import PyPDF2
    pdf_file = open(file, 'rb')
    read_pdf = PyPDF2.PdfFileReader(pdf_file)
    number_of_pages = read_pdf.getNumPages()
    page = read_pdf.getPage(0)
    page_content = page.extractText()
    print (page_content.encode('utf-8'))
    return page_content


       

def searchbasedonfilecontent(pattern):
    file=input('Enter the file in which pattern to be searched') 
    flag,path1=searchbasedonfilename(file)
    flag1=0
    print(path1)
    if flag and ('pdf' in path1):
        content=pdfreadder(path1)
        print(content)
        with open('tem.txt','w') as f1:
            f1.write(content)
        with open('tem.txt', 'r') as f2:
         for line in f2.readlines():
             match=re.findall(pattern, line)
             if(match):
                 flag1=1
                 with open("temp.txt",'w') as f:
                     f.write(line)
    elif flag :
       with open(path1, 'r') as f2:
         for line in f2.readlines():
             match=re.findall(pattern, line)
             if(match):
                 flag1=1
                 with open("temp.txt",'w') as f:
                     f.write(line)
                 
         if(flag1):
             print('Pattern found')
             return temp.txt
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
    for root,dirs,files in os.walk(path):
        c1=0
        c2=0
        c3=0
        c4=0
        count=[]
        count_files=[]
        for f1 in files:
            path1=os.path.join(root,f1)
            if f1.endswith(".txt") and (pattern in f1):
                c1=c1+1
                count_files.append(path1)
            if (f1.endswith(".img") or f1.endswith(".img") or f1.endswith(".img") or f1.endswith(".img"))and (pattern in f1):
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
        '''
        print("Total number text files in path {0} is :{1}".format(root,count[0]))
        print("Total number image files in path {0} is :{1}".format(root,count[1]))
        print("Total number pdf files in path {0} is :{1}".format(root,count[2]))
        print("Total number word documnet files in path {0} is :{1}".format(root,count[3]))
        '''
    return count_files 








 

options = {1: searchbasedonfilename,2:searchbasedonfilecontent,3:searchbasedondate,4:searchbasedonsize,5:searchbasedontypeoffile}    

print(Programs)

key = 'Y'

# ch=int(input('Enter the program Number :'))
while (key == 'Y') or (key == 'y') or (key == 'yes') or key == 'Yes' or key == 'YES':
    print("#####################################################\n")
    ch = int(input('Enter the Search criteria :'))
    pattern = input('Enter the pattern based on Search criteria :')
    
    print("\n")
    print("#####################################################\n")
    result=options[ch](pattern)
    print(result)
    key = input('Do you want to continue : Press Y/Yes/y/yes or N/No/no/n : ')

if key == 'N' or key == 'n' or key == 'no' or key == 'NO' or key == 'No':
    os.system('exit')

