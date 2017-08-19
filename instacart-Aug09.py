# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 10:07:03 2017

@author: rishi
"""


import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score
from time import gmtime, strftime
from sklearn import linear_model
#np.seterr(divide='ignore', invalid='ignore')


path1='C:/Manju-Documents/A-Zudemy/Machine Learning A-Z Template Folder/Datasets/Classification/Instacartdataset'

final=pd.DataFrame(data=None,index=[0],columns=['order_id','user_id','product_id','predicted','actual'])
cv_label=pd.DataFrame(data=None,index=[0],columns=['order_id','products','Products_actual'])

'''
df=pd.read_csv(path1+"/orders.csv")
df1=df[df['eval_set']=='train']
df2=df[df['eval_set']=='prior']
    
test=df[df['eval_set']=='test']

print('Added dept table...')
products=pd.read_csv(path1+'/products.csv')
departments=pd.read_csv(path1+'/departments.csv')
aisles=pd.read_csv(path1+'/aisles.csv')
'''
def unpickling():
   prior_train=pickle.load(open('prior_train.sav','rb'))
   prior_test=pickle.load(open('prior_test.sav','rb'))
   filename='model1.sav'
   clf=pickle.load(open(filename,'rb'))
   
   return prior_train,prior_test,clf


   
def Dumping_files(final,cv_label,clf,var):
   filename='testfinal.sav'
   pickle.dump(final,open(filename,'wb')) 
   filename='Testdump.sav'
   pickle.dump(cv_label,open(filename,'wb'))
   if var:
    filename='model1.sav'
    pickle.dump(clf,open(filename,'wb'))
  
   '''
   filename='prior_train.sav'
   pickle.dump(prior_train,open(filename,'wb'))
   filename='prior_test.sav'
   pickle.dump(prior_test,open(filename,'wb'))	
   '''

def load_csv(EDA=True):
    aisles = pd.read_csv(path1+'/aisles.csv', engine='c')
    print('Total aisles: {}'.format(aisles.shape[0]))
    aisles.head()
    
    dept = pd.read_csv(path1+'/departments.csv', engine='c')
    print('Total departments: {}'.format(dept.shape[0]))
    dept.head()
    
    products = pd.read_csv(path1+'/products.csv', engine='c')
    print('Total products: {}'.format(products.shape[0]))
    products.head()
    
    goods=pd.merge(pd.merge(products,dept,how='left',on='department_id'),aisles,how='left',on='aisle_id')
    goods.product_name = goods.product_name.str.replace(' ', '_').str.lower() 
    
    orders=pd.read_csv(path1+"/orders.csv",engine='c',dtype={'order_id': np.int32, 
                                                           'user_id': np.int32, 
                                                           'order_number': np.int32, 
                                                           'order_dow': np.int8, 
                                                           'order_hour_of_day': np.int8, 
                                                           'days_since_prior_order': np.float16})
    df1=orders[orders['eval_set']=='train']
    df2=orders[orders['eval_set']=='prior']
    
    test=orders[orders['eval_set']=='test']
    print('Loading train_csv')
    op_train_df=pd.read_csv(path1+'/order_products__train.csv',engine='c', 
                       dtype={'order_id': np.int32, 'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 'reordered': np.int8})
    #print('Loading prior_csv')
    op_prior_df=pd.read_csv(path1+'/order_products__prior.csv',engine='c', 
                       dtype={'order_id': np.int32, 'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 'reordered': np.int8})
    #merge1=pd.concat([op_train_df,op_prior_df])
    
    print('Loading merged files')
    
    train_df=pd.merge(df1,op_train_df,on='order_id',how='left')
    
    prior_df = pd.merge(df2,op_prior_df,on='order_id',how='left')
    
    prior_train=prior_df[prior_df['user_id'].isin(df1['user_id'].values)]
    prior_test=prior_df[prior_df['user_id'].isin(test['user_id'].values)]
    
    print('Spilting the dataframes merged files')
    if EDA:
        #Plot the product count department wise
        s=goods['product_id'].groupby(goods['department']).count()
        s.sort_values(ascending=False).plot(kind='bar', figsize=(12,5),title='Departments: Product #')
        
        #Plot the product count aisle wise
        s1=goods['product_id'].groupby(goods['aisle']).count()
        s1.sort_values(ascending=False).plot(kind='bar', figsize=(30,5),title='aisle: Product #')
        
        for i,e in enumerate(dept.department):
            s2=goods[goods.department==e].groupby(goods['aisle']).count()
            s2.plot(kind='bar', figsize=(12,5),title='deptment: '+e)
            
        #orders split by hour
        
        train_df['order_id'].groupby(train_df['order_hour_of_day']).count().plot(kind='bar')
        
        #orders split by dow
        train_df['order_id'].groupby(train_df['order_dow']).count().plot(kind='bar')
        #orders split by days_since_prior_order
        train_df['order_id'].groupby(train_df['days_since_prior_order']).count().plot(kind='bar')
        
        
        #orders split by hour
        
        train_df['reordered'].groupby(train_df['order_hour_of_day']).sum().plot(kind='bar')
        
        #orders split by dow
        train_df['reordered'].groupby(train_df['order_dow']).sum().plot(kind='bar')
        #orders split by days_since_prior_order
        train_df['reordered'].groupby(train_df['days_since_prior_order']).sum().plot(kind='bar')

    return train_df,test,prior_df,prior_train,prior_test,goods

	
def clean_data_hours_of_day(x):
  try:
   if(x>=0 and x<=7):
      return int(1)
   elif(x>7 and x<=14):
      return int(2)
   elif(x>14 and x<=20):
      return int(3)
   elif(x>20):
      return int(4)
  except TypeError:
      return None
	  
def clean_data_days_since_prior(x):
  
  try:
   if(x>0 and x<=10):
      return int(1)
   elif(x>10 and x<=20):
      return int(2)
   elif(x>20):
      return int(3)
  except TypeError:
      return None
	  
    	  
           


def getCountVar(compute_df,count_dict,var_name):
	
	count_list = []
	for index, row in compute_df.iterrows():
         name=row[var_name]
         key1=name
         if key1 in count_dict.keys():
             count_list.append(int(count_dict[key1]))
         else:
             count_list.append(0)
       
	return (count_list)

def getCountVar2(compute_df,count_dict,var_name1,var_name2):
	
	count_list = []
	for index, row in compute_df.iterrows():
         name1 = row[var_name1]
         name2 = row[var_name2]
         key1=(name1,name2)
         if key1 in count_dict.keys():
             count_list.append(int(count_dict[key1]))
         else:
             count_list.append(0)
	return (count_list)

def getCountVar3(compute_df,count_dict,var_name1,var_name2,var_name3):
    count_list = []
    for index, row in compute_df.iterrows():
         name1 = row[var_name1]
         name2 = row[var_name2]
         name3 = row[var_name3]
         key1=(name1,name2,name3)
         if key1 in count_dict.keys():
             count_list.append(int(count_dict[key1]))
         else:
             count_list.append(0)
         
    return (count_list)

def prob_prod_cart(compute_df,var_name1,var_name2):
	count_list = []
	for index, row in compute_df.iterrows():
         name1 = row[var_name1]
         name2 = row[var_name2]
         name3 = name1/name2
         count_list.append(name3)
	return count_list



def Feature_Engineering(smerge,goods):
    
    smerge=pd.merge(smerge,goods,on='product_id',how='left')
    
   
    
    
    smerge=smerge.fillna(0)
    #smerge1 = smerge.copy()
    #smerge = smerge[(smerge.order_number != 1) & (smerge.eval_set == 'prior')]
    smerge = smerge[smerge.order_number != 1]
    maxorder_userid=smerge['order_number'].groupby(smerge['user_id']).max()
    user_preferred_hours=smerge.groupby(['user_id'])['order_hour_of_day'].mean()
    Size_ordercartonordernum=smerge.groupby(['user_id','order_number'])['add_to_cart_order'].count()
    
    orderidReorder=smerge['reordered'].groupby(smerge['order_id']).sum()
    productidReorder=smerge['reordered'].groupby(smerge['product_id']).sum()
    useridReorder=smerge['reordered'].groupby(smerge['user_id']).sum()
    deptidReorder=smerge['reordered'].groupby(smerge['department_id']).sum()
    aisleidReorder=smerge['reordered'].groupby(smerge['aisle_id']).sum()
    orderweekdayReorder=smerge['reordered'].groupby(smerge['order_dow']).sum()
    orderhourReorder=smerge['reordered'].groupby(smerge['order_hour_of_day']).sum()
    ordersincepriorReorder=smerge['reordered'].groupby(smerge['days_since_prior_order']).sum()
    max_add_to_cart=smerge.groupby(['user_id'])['add_to_cart_order'].count()
	#repreated useridReorder
    reorderedcount=smerge.groupby(['user_id'])['reordered'].sum()
    reorderedproductcount=smerge.groupby(['user_id','product_id'])['reordered'].sum()
    
	
    reorderedproduct_order_num_count=smerge.groupby(['user_id','order_number'])['reordered'].sum()
    reorderedproduct_order_dow_count=smerge.groupby(['user_id','order_dow'])['reordered'].sum()
    reorderedproduct_order_hod_count=smerge.groupby(['user_id','order_hour_of_day'])['reordered'].sum()
    reorderedproduct_order_dsp_count=smerge.groupby(['user_id','days_since_prior_order'])['reordered'].sum()
    deptiduseridReorder=smerge.groupby(['user_id','department_id'])['reordered'].sum()
    aisleiduseridReorder=smerge.groupby(['user_id','aisle_id'])['reordered'].sum()
	
	##################################################################################
    productid_ordernumberReorder=smerge.groupby(['product_id','order_number'])['reordered'].sum()
    productid_dowReorder=smerge.groupby(['product_id','order_dow'])['reordered'].sum()
    productid_hodReorder=smerge.groupby(['product_id','order_hour_of_day'])['reordered'].sum()
    productid_dspReorder=smerge.groupby(['product_id','days_since_prior_order'])['reordered'].sum()
    productid_aisleReorder=smerge.groupby(['product_id','aisle_id'])['reordered'].sum()
	
   #############################################################################################################3
    
    productid_useridReorder=smerge.groupby(['product_id','user_id'])['reordered'].sum()
    productid_deptidReorder=smerge.groupby(['product_id','department_id'])['reordered'].sum()
    productid_deptid_priorReorder=smerge.groupby(['product_id','department_id','days_since_prior_order'])['reordered'].sum()
    productid_dept_id_hourReorder=smerge.groupby(['product_id','department_id','order_hour_of_day'])['reordered'].sum()
    productid_dept_id_dayReorder=smerge.groupby(['product_id','department_id','order_dow'])['reordered'].sum()
	
	##################################################################
    
    productid_ordernumberReorder=productid_ordernumberReorder.to_dict()
    productid_dowReorder=productid_dowReorder.to_dict()
    productid_hodReorder=productid_hodReorder.to_dict()
    productid_dspReorder=productid_dspReorder.to_dict()
    productid_aisleReorder=productid_aisleReorder.to_dict()
    var=['user_id','order_number','order_dow','order_hour_of_day','days_since_prior_order','product_id','department_id','aisle_id']
	     
    
    #ncount_order=orderidReorder.to_dict()
    ncount_product=productidReorder.to_dict()
    #ncount_user=useridReorder.to_dict()
    ncount_dept=deptidReorder.to_dict()
    ncount_aisle=aisleidReorder.to_dict()
    ncount_dow=orderweekdayReorder.to_dict()
    ncount_hour=orderhourReorder.to_dict()
    ncount_sinceprior=ordersincepriorReorder.to_dict()
    ncount_prod_dept=productid_deptidReorder.to_dict()
    ncount_prod_dept_prior=productid_deptid_priorReorder.to_dict()
    ncount_prod_dept_hour=productid_dept_id_hourReorder.to_dict()
    ncount_prod_dept_day=productid_dept_id_dayReorder.to_dict()
    count_deptwiseReorder=deptiduseridReorder.to_dict()
    count_aisleiduseridReorder=aisleiduseridReorder.to_dict()
    Size_ordercartonordernum=Size_ordercartonordernum.to_dict()
    user_preferred_hours=user_preferred_hours.to_dict()
    count_order_num=reorderedproduct_order_num_count.to_dict()
    count_order_dow=reorderedproduct_order_dow_count.to_dict()
    count_order_hod=reorderedproduct_order_hod_count.to_dict()
    count_order_dsp=reorderedproduct_order_dsp_count.to_dict()
    
    count_maxorder_userid=maxorder_userid.to_dict()
    count_max_add_to_cart=max_add_to_cart.to_dict()
    count_reorderedcount=reorderedcount.to_dict()
    count_reorderedproductcount=reorderedproductcount.to_dict()
    s1=smerge[smerge['reordered']==1].groupby(['user_id','product_id'])['days_since_prior_order'].sum()
    s1=s1.to_dict()
    
    #smerge=smerge1
	
    ncount1=[]
    ncount2=[]
    ncount3=[]
    ncount4=[]
    ncount5=[]
    ncount6=[]
    ncount7=[]
    ncount8=[]
    ncount9=[]
    
    ncount10=[]
    ncount11=[]
    ncount12=[]
    ncount13=[]
    ncount14=[]
	
	
    count_list1 = []
    count_list2 = []
    count_list3 = []
    count_list4 = []
    count_list5 = []
    count_list6 = []
    count_list7 = []
    count_list8 = []
    count_list9 = []
    count_list10 = []
    count_list11 = []
    count_list12 = []
    count_list13 = []
    
     
    print('Adding new features1 in FE.........\n\n')
    
    var=['user_id','order_number','order_dow','order_hour_of_day','days_since_prior_order','product_id','department_id','aisle_id']
    
    for index, row in smerge.iterrows():
         name1 = row[var[5]]
         
         key1=(name1)
         if key1 in ncount_product.keys():
             ncount1.append(int(ncount_product[key1]))
         else:
             ncount1.append(0)
         name1 = row[var[6]]
         
         key1=(name1)
         if key1 in ncount_dept.keys():
             ncount2.append(int(ncount_dept[key1]))
         else:
             ncount2.append(0)
         name1 = row[var[7]]
         
         key1=(name1)
         if key1 in ncount_aisle.keys():
             ncount3.append(int(ncount_aisle[key1]))
         else:
             ncount3.append(0)
         name1 = row[var[2]]
         
         key1=(name1)
         if key1 in ncount_dow.keys():
             ncount4.append(int(ncount_dow[key1]))
         else:
             ncount4.append(0)
         name1 = row[var[3]]
         
         key1=(name1)
         if key1 in ncount_hour.keys():
             ncount5.append(int(ncount_hour[key1]))
         else:
             ncount5.append(0)
		###################
         name1 = row[var[5]]
         name2 = row[var[6]]
         key1=(name1,name2)
         if key1 in ncount_prod_dept.keys():
             ncount6.append(int(ncount_prod_dept[key1]))
         else:
             ncount6.append(0)
         name1 = row[var[5]]
         name2 = row[var[6]]
         name3 = row[var[4]]
         key1=(name1,name2,name3)
         if key1 in ncount_prod_dept_prior.keys():
             ncount7.append(int(ncount_prod_dept_prior[key1]))
         else:
             ncount7.append(0)
         name1 = row[var[5]]
         name2 = row[var[6]]
         name3 = row[var[3]]
         key1=(name1,name2,name3)
         if key1 in ncount_prod_dept_hour.keys():
             ncount8.append(int(ncount_prod_dept_hour[key1]))
         else:
             ncount8.append(0)
         name1 = row[var[5]]
         name2 = row[var[6]]
         name3 = row[var[2]]
         key1=(name1,name2,name3)
         if key1 in ncount_prod_dept_day.keys():
             ncount9.append(int(ncount_prod_dept_day[key1]))
         else:
             ncount9.append(0)
         name1 = row[var[5]]
         name2 = row[var[1]]
         
         key1=(name1,name2)
         if key1 in productid_ordernumberReorder.keys():
             ncount10.append(int(productid_ordernumberReorder[key1]))
         else:
             ncount10.append(0)
         name1 = row[var[5]]
         name2 = row[var[2]]
         
         key1=(name1,name2)
         if key1 in productid_dowReorder.keys():
             ncount11.append(int(productid_dowReorder[key1]))
         else:
             ncount11.append(0)
         name1 = row[var[5]]
         name2 = row[var[3]]
         key1=(name1,name2)
         if key1 in productid_hodReorder.keys():
             ncount12.append(int(productid_hodReorder[key1]))
         else:
             ncount12.append(0)
         name1 = row[var[5]]
         name2 = row[var[4]]
         key1=(name1,name2)
         if key1 in productid_dspReorder.keys():
             ncount13.append(int(productid_dspReorder[key1]))
         else:
             ncount13.append(0)
         name1 = row[var[5]]
         name2 = row[var[7]]
         key1=(name1,name2)
         if key1 in productid_aisleReorder.keys():
             ncount14.append(int(productid_aisleReorder[key1]))
         else:
             ncount14.append(0)
		 
		 ###########################################3
         name1 = row[var[0]]
         name2 = row[var[1]]
         key1=(name1,name2)
         if key1 in Size_ordercartonordernum.keys():
             count_list1.append(int(Size_ordercartonordernum[key1]))
         else:
             count_list1.append(0)
             
         
        
         key1=(name1,name2)
         if key1 in count_order_num.keys():
             count_list2.append(int(count_order_num[key1]))
         else:
             count_list2.append(0)
         
         key1=(name1)
         if key1 in user_preferred_hours.keys():
             count_list3.append(int(user_preferred_hours[key1]))
         else:
             count_list3.append(0)
             
         
         name2 = row[var[2]]
         key1=(name1,name2)
         if key1 in count_order_dow.keys():
             count_list4.append(int(count_order_dow[key1]))
         else:
             count_list4.append(0)
             
         name1
         name2 = row[var[3]]
         key1=(name1,name2)
         if key1 in count_order_hod.keys():
             count_list5.append(int(count_order_hod[key1]))
         else:
             count_list5.append(0)
             
         name1 = row[var[0]]
         name2 = row[var[4]]
         key1=(name1,name2)
         if key1 in count_order_dsp.keys():
             count_list6.append(int(count_order_dsp[key1]))
         else:
             count_list6.append(0)
         
         
         key1=(name1)
         if key1 in count_maxorder_userid.keys():
             count_list7.append(int(count_maxorder_userid[key1]))
         else:
             count_list7.append(0)
         
         
         if key1 in count_max_add_to_cart.keys():
             count_list8.append(int(count_max_add_to_cart[key1]))
         else:
             count_list8.append(0)
             
         
         if key1 in count_reorderedcount.keys():
             count_list9.append(int(count_reorderedcount[key1]))
         else:
             count_list9.append(0)
             
        
         name2 = row[var[5]]
         key1=(name1,name2)
         if key1 in count_reorderedproductcount.keys():
             count_list10.append(int(count_reorderedproductcount[key1]))
         else:
             count_list10.append(0)
             
         
         name2 = row[var[6]]
         key1=(name1,name2)
         if key1 in count_deptwiseReorder.keys():
             count_list11.append(int(count_deptwiseReorder[key1]))
         else:
             count_list11.append(0)
             
         
         name2 = row[var[7]]
         key1=(name1,name2)
         if key1 in count_aisleiduseridReorder.keys():
             count_list12.append(int(count_aisleiduseridReorder[key1]))
         else:
             count_list12.append(0)
        
         name2 = row[var[5]]
         key1=(name1,name2)
         if key1 in s1.keys():
             count_list13.append(int(s1[key1])*10)
         else:
             count_list13.append(0)
             
	

    
    
    smerge['Size_ordercartonordernum']=count_list1
    smerge['order_num_count']=count_list2
    smerge['user_preferred_hours']=count_list3
    
    smerge['order_hour_of_day']=smerge['order_hour_of_day'].fillna(smerge['user_preferred_hours'])
    
    smerge['order_dow_count']=count_list4
    smerge['order_hod_count']=count_list5
    smerge['order_dsp_count']=count_list6
    smerge['maxorder_userid']=count_list7
    smerge['count_max_add_to_cart']=count_list8
    smerge['count_reorderedcount']=count_list9
    smerge['count_reorderedproductcount']=count_list10
    smerge['deptiduseridReorder']=count_list11
    smerge['aisleiduseridReorder']=count_list12
    smerge['dsp_sum']=count_list13
    smerge['count1']=ncount1
    smerge['count2']=ncount2
    smerge['count3']=ncount3
    smerge['count4']=ncount4
    smerge['count5']=ncount5
    smerge['count6']=ncount6
    smerge['count7']=ncount7
    smerge['count8']=ncount8
    smerge['count9']=ncount9
    smerge['count10']=ncount10
    smerge['count11']=ncount11
    smerge['count12']=ncount12
    smerge['count13']=ncount13
    smerge['count14']=ncount14
    
    print('Adding new features2 in FE.........\n\n')
    
    #smerge['P(Reordered|order_num)']=smerge.loc[:,['order_num_count','Size_ordercartonordernum']].apply((lambda x : x[0]/x[1]) ,axis=1)
    smerge['P(Reordered)']=smerge.loc[:,['count_reorderedcount','count_max_add_to_cart']].apply((lambda x : x[0]/x[1]) ,axis=1)
    #smerge['P(A)']=smerge.loc[:,['deptiduseridReorder','count_max_add_to_cart']].apply((lambda x : (x[0])/x[1]) ,axis=1)
    smerge['P(A | Reordered)']=smerge.loc[:,['count_reorderedproductcount','count_reorderedcount']].apply((lambda x : x[0]/x[1]) ,axis=1)
    #smerge['P(Reordered|A)']=smerge.loc[:,['P(A | Reordered)','P(Reordered)','P(A)']].apply((lambda x : (x[0]*x[1])/x[2]) ,axis=1)
    
    smerge['Size_cart']=smerge.loc[:,['count_max_add_to_cart','maxorder_userid']].apply((lambda x : int(x[0]/x[1])) ,axis=1)
    smerge['Size_reordercart']=smerge.loc[:,['count_reorderedcount','maxorder_userid']].apply((lambda x : int(x[0]/x[1])) ,axis=1)
    smerge['P(Reorderedhour)']=smerge.loc[:,['order_hod_count','count_reorderedcount']].apply((lambda x : x[0]/x[1]) ,axis=1)
    smerge['P(Reordereddow)']=smerge.loc[:,['order_dow_count','count_reorderedcount']].apply((lambda x : x[0]/x[1]) ,axis=1)
    smerge['P(Reordereddsp)']=smerge.loc[:,['order_dsp_count','count_reorderedcount']].apply((lambda x : x[0]/x[1]) ,axis=1)
    smerge['P(Reordereddspordernum)']=smerge.loc[:,['order_num_count','count_reorderedcount']].apply((lambda x : x[0]/x[1]) ,axis=1)
   
    smerge['rank_dsp']=smerge.loc[:,['dsp_sum','count_reorderedproductcount']].apply((lambda x : x[0]/x[1] if (x[1] >0 and x[0]>0) else 60) ,axis=1)
    
    
    return smerge
	
def train_model(smerge):
    
    X1=smerge.copy()
    X1=X1.fillna(0)
    X_train=X1
    y_train=X_train['reordered']
    X_train=X_train.drop(['department','aisle','eval_set','product_name','reordered'],axis=1)
    X_train=X_train.drop(['order_id','user_id','order_number','add_to_cart_order','product_id','aisle_id','department_id'],axis=1)
    
    #clf=RandomForestClassifier(n_estimators=50,random_state=44)
    #clf = linear_model.SGDClassifier()
    clf=linear_model.SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
        eta0=0.0, fit_intercept=True, l1_ratio=0.15,
        learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
        penalty='l2', power_t=0.5, random_state=None, shuffle=True,
        verbose=0, warm_start=False)
    clf = clf.fit(X_train,y_train)
    #importance=clf.feature_importances_
    return X_train,clf

def test_model(smerge,clf):
    X1=smerge.copy()
    
    test=smerge[smerge['eval_set']=='test']
    X1=X1.fillna(0)
    
    
    X_test=X1[X1['eval_set']=='test']
    y_test=X_test['reordered']
    X_test=X_test.drop(['department','aisle','eval_set','product_name','reordered'],axis=1)
    X_test=X_test.drop(['order_id','user_id','order_number','add_to_cart_order','product_id','aisle_id','department_id'],axis=1)
    
    y_pred = clf.predict(X_test)
   
    score=log_loss(y_test,y_pred) 
    print('Logloss score of Random forestclassifier is {0}'.format(score))
    
    return X_test,y_pred,y_test,test

    
def submission(final,cv_label,test):
    
    f=test.loc[:,['order_id','user_id','product_id']]
    f['predicted']=y_pred
    f['actual']=y_test 
    final=pd.concat([final,f])
    final=final.dropna()
    
    
	 
    final_pred=final[final['predicted']==1]
    g1=final_pred['product_id'].groupby(final_pred['order_id'])
    count2={}
    for i, name in g1:
        s=list(set(name))
        str1 = ' '.join(str(int(e)) for e in s)
        count2[i]=str1
          
    final_act=final[final['actual']==1]
    g2=final_act['product_id'].groupby(final_act['order_id'])
    count3={}
    for i, name in g2:
        s=list(set(name))
        str1 = ' '.join(str(int(e)) for e in s)
        count3[i]=str1
          
    
    
    cv_labels_df=pd.DataFrame(data=test.order_id.unique(),columns=['order_id'])
    #cv_labels_df=test[['order_id']]
    
    count_list1 = []
    count_list2 = []
    for index, row in cv_labels_df.iterrows():
        name=row['order_id']
        count_list1.append(count2.get(name,0))
        count_list2.append(count3.get(name,0))
       
        
    cv_labels_df['products']=count_list1
    cv_labels_df['Products_actual']=count_list2
    cv_label=pd.concat([cv_labels_df,cv_label])
    return cv_label

def csvformat(cv_label):
    cv_label1=cv_label.drop(['Products_actual'],axis=1)
    cv_label1=cv_label1.dropna()
    cv_label1['order_id']=cv_label1['order_id'].astype(int)
    #cv_label1=cv_label1.dropna()
    cv_label1.to_csv(path1+ '/output.csv',index=False)
    return cv_label1
    
def splitdata_test(sprior_df_test,test1):
    
    #sprior_df_train=prior_df1[prior_df1['user_id'].isin(trian_df1['user_id'].values)]
    #sprior_df_test=prior_df1[prior_df1['user_id'].isin(test1['user_id'].values)]
    
    count2={}
    g1=sprior_df_test.groupby(sprior_df_test['user_id'])
    s=pd.DataFrame(data=None,index=[0],columns=['user_id','product_id','reordered','add_to_cart_order'])
    for i, name in g1:
        name=name[name.order_number==(name['order_number'].max()-1)]
        #name=name[name['reordered'].sum()==g2]
        count2[i]=name
        s1=pd.DataFrame({'user_id':[i]*len(count2[i]),'product_id':name['product_id'],'reordered':name['reordered'],'add_to_cart_order':name['add_to_cart_order']})
        s=pd.concat([s,s1])
    s=s.dropna()
    stest=pd.merge(test1,s,on='user_id',how='left')
    smerge_stest=pd.concat([stest,sprior_df_test],ignore_index=True)
    return smerge_stest

def splitdata_train(sprior_df_train,strain1):
    '''
    count3={}
    g2=sprior_df_train['product_id'].groupby(sprior_df_train['user_id'])
    t=pd.DataFrame(data=None,index=[0],columns=['user_id','product_id','add_to_cart_order'])
    for i, name in g2:
        count3[i]=list(set(name))
        t1=pd.DataFrame({'user_id':[i]*len(count3[i]),'product_id':count3[i],'add_to_cart_order':len(count3[i])})
        t=pd.concat([t,t1])
    t=t.dropna()
    strian_df_tmp=strian1.drop(['product_id','add_to_cart_order','reordered'],axis=1)
    strian_df_tmp=strian_df_tmp.drop_duplicates()
    h=strian1.groupby(['user_id','product_id'])['reordered'].count()
    h_count=h.to_dict()
    strain=pd.merge(t,strian_df_tmp,on='user_id',how='left')
    strain['reordered']=getCountVar2(strain,h_count,'user_id','product_id')
    '''
    
    
    smerge_strain=pd.concat([sprior_df_train,strain1],ignore_index=True)
    #smerge_strain['order_id'].groupby(smerge_strain['user_id']).max()
    
    
    return smerge_strain
    
    
    
def clean_data(smerge):
    #values=smerge.apply(lambda x : sum(x.isnull()))
    
    smerge['order_hour_of_day']=smerge['order_hour_of_day'].apply(clean_data_hours_of_day)
    smerge['days_since_prior_order']=smerge['days_since_prior_order'].apply(clean_data_hours_of_day)
    d1=pd.get_dummies(smerge['order_hour_of_day'],prefix='order_hour_of_day')
    d2=pd.get_dummies(smerge['days_since_prior_order'],prefix='days_since_prior_order')
    d3=pd.get_dummies(smerge['order_dow'],prefix='order_dow')
    smerge=pd.concat([smerge,d1,d2,d3],axis=1)
    smerge=smerge.drop(['order_hour_of_day','days_since_prior_order','order_dow'],axis=1)
    return smerge


def calculate_f1score(labels, preds):
    labels = labels.split(' ')
    preds = preds.split(' ')
    rr = (np.intersect1d(labels, preds))
    precision = np.float(len(rr)) / len(preds)
    recall = np.float(len(rr)) / len(labels)
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return (precision, recall, 0.0)
    return (precision, recall, f1)

def mean_f1score(cv_label):
    res = list()
    cv_label['products']=cv_label['products'].astype(str)
    cv_label['Products_actual']=cv_label['Products_actual'].astype(str)
    
    for entry in cv_label.itertuples():
        res.append(calculate_f1score(entry[2], entry[3]))
    res = pd.DataFrame(np.array(res), columns=['precision', 'recall', 'f1'])
    return res['f1'].mean()	


    
if __name__ == '__main__':
   if False: 
       print("StartTime: "+ strftime("%Y-%m-%d %H:%M:%S", gmtime()))
       train_df,test,prior_df,prior_train,prior_test,goods=load_csv()
   
   if False:  
    print('Performing the unpickling............')   
    prior_train,prior_test,clf=unpickling()
   #cv_label=pd.DataFrame(data=None,index=[0],columns=['order_id','Products_pred','Products_actual'])
   indexes_train = np.linspace(0, len(prior_train), num=100, dtype=np.int32)
   indexes_test = np.linspace(0, len(prior_test), num=10, dtype=np.int32)
   
   
   
   train=False
   testing=True
   for i in range(0,len(indexes_train)-1):
   #for i in range(0,1):
       if train:
           print('Number of iteration value of is: '+str(i))
           
           prior_train1=prior_train.loc[indexes_train[i]:indexes_train[i+1], :]
           #strian_df1=train_df[index][(train_df[index]['user_id'] >=k) & (train_df[index]['user_id'] <=l)]
           k=prior_train1['user_id'].min()
           l=prior_train1['user_id'].max()
           strian1=train_df[(train_df['user_id'] >=k) & (train_df['user_id'] <= l+1)]
           
           
        
           print('Performing the spiltdat............')
          
          
           
           strain1=splitdata_train(prior_train1,strian1)
           print('Performing the cleaning............')
           
           
           #strain=clean_data(strain1)
           
           print('Performing the Feature_Engineering..........')
           strain=Feature_Engineering(strain1,goods)
           train_data=clean_data(strain)
           print('Performing the train_model..........')
           
           
           X_train,clf=train_model(train_data)
           
           
           print("EndTime: "+ strftime("%Y-%m-%d %H:%M:%S", gmtime()))
   for i in range(0,len(indexes_test)-1):    
       if testing:
           print('Number of iteration value of is: '+str(i))
           
           
           prior_test1=prior_test.loc[indexes_test[i]:indexes_test[i+1], :]
           #strian_df1=train_df[index][(train_df[index]['user_id'] >=k) & (train_df[index]['user_id'] <=l)]
           k=prior_test1['user_id'].min()
           l=prior_test1['user_id'].max()
           test1=test[(test['user_id'] >=k) & (test['user_id'] <= l)]
           
           
        
           print('Performing the spiltdat............')
           '''
           test1=test[(test['user_id'] >k) & (test['user_id'] <=l)]
           prior_test1=prior_test[(prior_test['user_id'] >k) & (prior_test['user_id'] <=l)]
           '''
           print('Performing the spiltdat............')
           stest1=splitdata_test(prior_test1,test1)
          
           print('Performing the Feature_Engineering..........')
           stest=Feature_Engineering(stest1,goods)
           print('Performing the cleaning............')
           test_data=clean_data(stest)
           print('Performing the testing_model..........')
           X_test,y_pred,y_test,tested=test_model(test_data,clf)
           print('Performing the submission..........')
           cv_label=submission(final,cv_label,tested)
           print('Performing the mean_f1score..........')
           f1_score=mean_f1score(cv_label)
           k=l
           l=l+ss
           
           print("EndTime: "+ strftime("%Y-%m-%d %H:%M:%S", gmtime()))
       
   print('Performing the Dumping_files..........')
   Dumping_files(final,cv_label,clf,train)
   print('Performing the mean_f1score..........')
   f1_score=mean_f1score(cv_label)
   output=csvformat(cv_label)
   
   print("EndTime: "+ strftime("%Y-%m-%d %H:%M:%S", gmtime()))
       


    
   



