
	

import os
import re
import fnmatch
path='C:\\Users\\rishi\\Desktop\\Search_Engine'
Programs = '''
1.Search based on filename
2.Search based on file title
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
       



       

def searchbasedonfilecontent(pattern):
    file=input('Enter the file in which pattern to be searched') 
    flag,path1=searchbasedonfilename(file)
    if flag:
        with open(path1, 'r') as f:
            for line in f:
                if pattern in line:
                    #for i in range(8):
                    print (line)
    else:
        print('File not found')
'''
def searchbasedondate():
    for i,j,k in os.walk(path):
        c1=0,c2=0,c3=0,c4=0
        for f1 in k:
            if f1.endswith(".txt"):
                c1=c1+1
            if f1.endswith(".img"):
                c2=c2+1
            if f1.endswith(".pdf"):
                c3=c3+1
            if f1.endswith(".doc"):
                c4=c4+1
def searchbasedonsize():
    for i,j,k in os.walk(path):
        c1=0,c2=0,c3=0,c4=0
        for f1 in k:
            if f1.endswith(".txt"):
                c1=c1+1
            if f1.endswith(".img"):
                c2=c2+1
            if f1.endswith(".pdf"):
                c3=c3+1
            if f1.endswith(".doc"):
                c4=c4+1
				
def searchbasedontypeoffile():
    for i,j,k in os.walk(path):
        c1=0,c2=0,c3=0,c4=0
        for f1 in k:
            if f1.endswith(".txt"):
                c1=c1+1
            if f1.endswith(".img"):
                c2=c2+1
            if f1.endswith(".pdf"):
                c3=c3+1
            if f1.endswith(".doc"):
                c4=c4+1
       
        count.append(c1)
        count.append(c2)
        count.append(c3)
        count.append(c4)
        print("Total number text files in path {0} is :{1}".format(i,count[0]))
        print("Total number image files in path {0} is :{1}".format(i,count[1]))
        print("Total number pdf files in path {0} is :{1}".format(i,count[2]))
        print("Total number word documnet files in path {0} is :{1}".format(i,count[3]))
            
'''







#options = {1: searchbasedonfilename, 2:searchbasedonfilecontent , 3: searchbasedondate, 4:searchbasedonsize, 5:searchbasedontypeoffile} 

options = {1: searchbasedonfilename,2:searchbasedonfilecontent}    

print(Programs)

key = 'Y'

# ch=int(input('Enter the program Number :'))
while (key == 'Y') or (key == 'y') or (key == 'yes') or key == 'Yes' or key == 'YES':
    print("#####################################################\n")
    ch = int(input('Enter the Search criteria :'))
    pattern = input('Enter the pattern based on Search criteria :')
    print("\n")
    print("#####################################################\n")
    options[ch](pattern)
    key = input('Do you want to continue : Press Y/Yes/y/yes or N/No/no/n : ')

if key == 'N' or key == 'n' or key == 'no' or key == 'NO' or key == 'No':
    os.system('exit')

