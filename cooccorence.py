import pandas as pd

file_name = './data/multilabel.txt'

data_df = pd.read_csv(file_name, 
                                sep='\t', 
                                header=None, 
                                names=['code', 'sub_code', 'abstract', 'train_or_test'])
data_df['sub_code'] = data_df['sub_code'].astype('str')

group_count = data_df.groupby(['code','sub_code']).count().sort_values(ascending=False,by='abstract').reset_index()



CODE_TO_INDEX = {'A01':0, 'A02':1, 'A03':2, 'A04':3, 'A05':4,
                'B01':5, 'B02':6, 'B03':7, 'B04':8, 'B05':9,
                'B06':10, 'B07':11, 'B08': 12, 'C01':13, 'C02':14, 
                'C03':15, 'C04':16, 'C05':17, 'C06':18, 'C07':19,
                'C08':20, 'C09':21, 'C10':22, 'C11':23, 'C12':24,
                'C13':25, 'C14':26, 'C15':27, 'C16':28, 'C17':29, 
                'C18':30, 'C19':31, 'C20':32, 'C21':33, 'D01':34, 
                'D02':35, 'D03':36, 'D04':37, 'D05':38, 'D06':39, 
                'D07':40, 'E01':41, 'E02':42, 'E03':43, 'E04':44, 
                'E05':45, 'E06':46, 'E07':47, 'E08':48, 'E09':49, 
                'F01':50, 'F02':51, 'F03':52, 'F04':53, 'F05':54, 
                'F06':55, 'G01':56, 'G02':57, 'G03':58, 'G04':59, 
                'H01':60, 'H02':61, 'H03':62, 'H04':63, 'H05':64, 
                'H06':65, 'H07':66, 'H08':67, 'H09':68, 'H10':69, 
                'H11':70, 'H12':71, 'H13':72, 'H14':73, 'H15':74, 
                'H16':75, 'H17':76, 'H18':77, 'H19':78, 'H20':79,
                'H21':80, 'H22':81, 'H23':82, 'H24':83, 'H25':84, 
                'H26':85, 'H27':86, 'H28':87, 'H29':88, 'H30':89, 
                'H31':90}

pattern_num = 50
f = open('./cooccur.txt', 'a')
print(group_count.head())
for i in range(pattern_num):
    discipline1 = CODE_TO_INDEX[group_count.iloc[i][0]]
    discipline2 = CODE_TO_INDEX[group_count.iloc[i][1]]
    count = group_count.iloc[i][2]
    f.write(str(discipline1) + ' ' + str(discipline2) + ' ' + str(count) + '\n')

f.close()