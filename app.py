import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#my_conn=create_engine("mysql://root:@localhost/test")
import pickle
with open('model_pkl','rb') as f:
   model=pickle.load(f)

pd.set_option('display.max_colwidth',-1)
# round off function
curr=['BTC-INR','ETH-INR','USDT-INR','BNB-INR','USDC-INR','SOL-INR','XRP-INR','ADA-INR','DOGE-INR','AVAX-INR','DOT-INR','SHIB-INR','CRO-INR','MATIC-INR','DAI-INR','NEAR-INR','LTC-INR','TRX-INR','BCH-INR','LINK-INR','ATOM-INR','FTT-INR','XLM-INR','ALGO-INR','XMR-INR','ETC-INR','FIL-INR','HBAR-INR','VET-INR','MANA-INR','EGLD-INR','SAND-INR','THETA-INR','XTZ-INR','RUNE-INR','CAKE-INR','EOS-INR','DFI-INR','AAVE-INR','AXS-INR','FTM-INR','ZEC-INR','FLOW-INR','HNT-INR','BTT-INR','MKR-INR','WAVES-INR','BSV-INR','TUSD-INR','STX-INR','NEO-INR','KSM-INR','QNT-INR','CHZ-INR','CELO-INR','ZIL-INR','ENJ-INR','LRC-INR','DASH-INR','BAT-INR']
feature=['Open','High','Low','Close']
pred_Curr=['BTC-INR','ETH-INR','BNB-INR']
curr=pd.Series(data=curr)

def round_off(value):
    
    
    if value>1:
        
        return float(round(value,2))
    else:
        return float(round(value,8)) 

def round_off2(value):
    
    return float(round(value,3))
    


st.title( 'COINDASHER')
st.sidebar.image('logo.png',width=100)
st.write('''*Cryptocurrency*  price  web based application .''')


st.header('**Cryptocurrencies Prices**')


#df = pd.read_json('https://api.binance.com/api/v3/ticker/24hr')
df=pd.read_json('https://api.coingecko.com/api/v3/coins/markets?vs_currency=inr&order=market_cap_desc&per_page=100&page=1&sparkline=false')
df2=pd.read_json('https://api.coingecko.com/api/v3/coins/markets?vs_currency=inr&order=market_cap_desc&per_page=100&page=1&sparkline=false')

# creating the matrix 
col1,col2,col3=st.columns(3)


# sidebar of the application

with st.sidebar.header('''Selected Cryptocurrencies'''):
    first_select=st.sidebar.selectbox('First',df.id,list(df.id).index('bitcoin')) 
    second_select=st.sidebar.selectbox('Second',df.id,list(df.id).index('wrapped-bitcoin'))
    third_select=st.sidebar.selectbox('Third',df.id,list(df.id).index('ethereum'))
    fourth_select=st.sidebar.selectbox('Fourth',df.id,list(df.id).index('binancecoin'))
    fifth_select=st.sidebar.selectbox('Fifth',df.id,list(df.id).index('bitcoin-cash'))
    sixth_select=st.sidebar.selectbox('Sixth',df.id,list(df.id).index('solana'))
    seventh_select=st.sidebar.selectbox('Seventh',df.id,list(df.id).index('dash'))
    eighth_select=st.sidebar.selectbox('Eigth',df.id,list(df.id).index('staked-ether'))
    ninth_select=st.sidebar.selectbox('Ninth',df.id,list(df.id).index('maker'))
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.markdown('---------------------------------')
    st.sidebar.write('')
    st.sidebar.write('')

with st.sidebar.header('''Visual Trend Of Cryptocurrencies'''):
    selected_currency=st.sidebar.selectbox('Select Currency',curr,list(curr).index('BTC-INR'))
    selected_feature=st.sidebar.selectbox('Select Particular Feature',feature,3)
    st.sidebar.write('')
    #selected_graph=st.sidebar.radio('Choose The Type Of Chart Element',('line_chart','bar_chart'))
    st.sidebar.subheader('Comparsion Between Currencies')
    st.sidebar.write('')
    first_comparision_currency=st.sidebar.selectbox('Select First Currency',curr,list(curr).index('BTC-INR'))
    st.sidebar.caption('------------------------------VS--------------------------------')
    second_comparision_currency=st.sidebar.selectbox('Select Second Currency',curr,list(curr).index('ETH-INR'))
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.markdown('---------------------------------')
    st.sidebar.write('')
    

with st.sidebar.title('Cryptocurrencies Forcasting'):
    predected_currency=st.sidebar.selectbox('Select Currency For Predition',pred_Curr)
    
    number_day_pred=st.sidebar.slider("Pick Number Of Days For Prediction",1,50)
    st.sidebar.info("Please Keep It Low For Better Output")


    

# select the particular currency in a form of the dataframe

row1_df=df[df.id==first_select]
row2_df=df[df.id==second_select]
row3_df=df[df.id==third_select]
row4_df=df[df.id==fourth_select]
row5_df=df[df.id==fifth_select]
row6_df=df[df.id==sixth_select]
row7_df=df[df.id==seventh_select]
row8_df=df[df.id==eighth_select]
row9_df=df[df.id==ninth_select]


# round off price of the paricular currency

first_price=row1_df.current_price
second_price=row2_df.current_price
third_price=row3_df.current_price
fourth_price=row4_df.current_price
fifth_price=row5_df.current_price
sixth_price=row6_df.current_price
seventh_price=row7_df.current_price
eighth_price=row8_df.current_price
ninth_price=row9_df.current_price


# price changes in the percentage of currency

first_currency_change=f'{round_off(float(row1_df.price_change_percentage_24h.values))}%'
second_currency_change=f'{round_off(float(row2_df.price_change_percentage_24h.values))}%'
third_currency_change=f'{round_off(float(row3_df.price_change_percentage_24h.values))}%'
fourth_currency_change=f'{round_off(float(row4_df.price_change_percentage_24h.values))}%'
fifth_currency_change=f'{round_off(float(row5_df.price_change_percentage_24h.values))}%'
sixth_currency_change=f'{round_off(float(row6_df.price_change_percentage_24h.values))}%'
seventh_currency_change=f'{round_off(float(row7_df.price_change_percentage_24h.values))}%'
eighth_currency_change=f'{round_off(float(row8_df.price_change_percentage_24h.values))}%'
ninth_currency_change=f'{round_off(float(row9_df.price_change_percentage_24h.values))}%'

# creating the matrix of price and changes 

col1.metric(first_select,first_price,first_currency_change)
col2.metric(second_select,second_price,second_currency_change)
col3.metric(third_select,third_price,third_currency_change)
col1.metric(fourth_select,fourth_price,fourth_currency_change)
col2.metric(fifth_select,fifth_price,fifth_currency_change)
col3.metric(sixth_select,sixth_price,sixth_currency_change)
col1.metric(seventh_select,seventh_price,seventh_currency_change)
col2.metric(eighth_select,eighth_price,eighth_currency_change)
col3.metric(ninth_select,eighth_price,ninth_currency_change)


btc_data=yf.Ticker(selected_currency)
#btc_data.info
btchis=btc_data.history(period="max")
btc=yf.download(selected_currency,start=pd.to_datetime('today'),end=pd.to_datetime('today'))
st.write('--------------------------------------------------------')
st.write('')
st.title('Visual Repersentation')
st.write('')
st.write('')
str=(df[df['symbol']==(selected_currency[:-4]).lower()].image.item())
#st.write(str)
st.image(str,width=60)
#st.write(type(str))
#st.write(type(str.encode()))
st.table(btc)
st.bar_chart(getattr(btchis,selected_feature))
   
st.write('')
st.write('')
st.write('')
col7,col8,col9=st.columns(3)
with col8:
    st.subheader('COMPARISION')


st.write('')
st.write('')
col4,col5,col6=st.columns(3)
first_data=yf.Ticker(first_comparision_currency)
second_data=yf.Ticker(second_comparision_currency)
first_data=yf.download(first_comparision_currency,start=pd.to_datetime('today'),end=pd.to_datetime('today'))
second_data=yf.download(second_comparision_currency,start="2022-4-30",end="2022-4-30")
with col4:
    st.write(first_comparision_currency)
    val=float(first_data.Open.values)
    st.write('Open :',val)
    val=float(first_data.High.values)
    st.write('High :',val)
    val=float(first_data.Low.values)
    st.write('Low :',val)
    val=float(first_data.Close.values)
    st.write('Close :',val)
with col5:
    st.text("  ")
with col6:
    st.write(second_comparision_currency)
    val=float(second_data.Open.values)
    st.write('Open :',val)
    val=float(second_data.High.values)
    st.write('High :',val)
    val=float(second_data.Low.values)
    st.write('Low :',val)
    val=float(second_data.Close.values)
    st.write('Close :',val)
st.write('')
st.write('')
dropdown=st.multiselect('Pick Your Currencies',curr,default=['BTC-INR','ETH-INR'])
star=st.date_input('Start',value=pd.to_datetime('2022-5-1'))
end=st.date_input('End',value=pd.to_datetime('today'))
features_for_comp=st.selectbox('Select Feature',feature,0)

def cummilative_return(df):
    rel=df.pct_change()
    cumm=(1+rel).cumprod()-1
    cumm =cumm.fillna(0)
    return cumm

if len(dropdown) >0:
    #df=yf.download(dropdown,star,end)[features_for_comp]
    df=cummilative_return(yf.download(dropdown,star,end)[features_for_comp])
    
    st.line_chart(df)

# for creating a dataset of list type data crypto history
def create_dataset(dataset,timestamp):
  datax,datay=[],[]
  for i in range(len(dataset)-timestamp-1):
    a=dataset[i:(i+timestamp),0]
    datax.append(a)
    datay.append(dataset[i+timestamp])
  return np.array(datax),np.array(datay)

# prediciting the unseen data
def pred_for_n_days(input_100,x_input,n):
    lst_out=[]
    n_step=100
    nextnumberofdays=n
    i=0
    while(i<nextnumberofdays) :
        if (len(input_100)>160):
            x_input=np.array(input_100[1:])
            print("{} days input {}".format(i,x_input))
            x_input=x_input .reshape(1,-1)
            x_input=x_input .reshape((1,n_step,1))
            yhat=model.predict(x_input)
            print("{} days output {}".format(i,yhat))
            input_100.extend(yhat[0].tolist())
            input_100=input_100[1: ]
            lst_out.extend(yhat.tolist())
            i=i+1
        else:
            x_input=x_input .reshape(1,n_step,1)
            yhat=model.predict (x_input)
            input_100.extend(yhat[0].tolist())
            print(len(input_100) )
            lst_out.append(yhat.list())
            i=i+1
            print(lst_out)  
            return lst_out


st.write('')
st.write('')
st.write('--------------------------------------------------------')
# neural network
st.write('')
st.write('')
st.write('')
st.title('CRYPTOCURRENCIES PREDITIONS')
st.write('')
st.write('')
st.write('')
currency_name=yf.Ticker(predected_currency)
data=currency_name.history(period='max')
data=np.array(data)
data=data.reshape(-1,1)
scaler=MinMaxScaler(feature_range=(0,1))
data=scaler.fit_transform(data)
x_data,y_data=create_dataset(data,100)
list_out=[]
list_out=model.predict(x_data)


x_input=data[ (data. shape[0]-101):, :].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

lis=pred_for_n_days(temp_input,x_input, number_day_pred)

entire_pred=np.concatenate((list_out, lis), axis=0)

entire_pred=scaler.inverse_transform(entire_pred)
y_data=scaler.inverse_transform(y_data)
first=pd.Series(entire_pred.ravel())
second=pd.Series(y_data.ravel())
demo=pd.DataFrame([ first, second]).T
demo.columns=['Predicited Data', 'Original Data']
st.line_chart(demo)

 

#plt.plot(y_data)
#plt.plot(list_out)
#st.write(data.reset_index()['Close'])
#plt.plot(data.reset_index()['Close'])


with st.expander("Description Of Model"):
    st.subheader('Model Description')
    st.write('''Above deep learning model is use to predict the three major cryptocurrencies.The 
    correlation between the prices of these cryptocurrencies is so high that is why this model 
    trained only bitcoin and this particular model can be used for rest of the currencies.''')
    st.write('''This model is trained only using the closing data of the particular currency
    .We use closing data because it is the most suitable data that would helps us to make good
    model for the predition''')
    





result={
    'id':[1,2,3,4,5,6,7,8],
    'name':['abhinish','nishu','aabhi','anil','abhisek','deep','har','cj']
}

demo=pd.DataFrame(data=result)
show1,show2=st.columns((2,2))

with show1:
    df

#with show2:
#    st.write('')
#    st.write('')
#    st.write('')
#    if st.button(label="Store data"):
#        demo.to_sql(con=my_conn,name="demo",if_exists='replace')
#        st.info("DATA IS STORED")
#    else :
#        st.info('CLICK HERE TO STORE THE DATA')
#

st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.header('**CRYPTOCURRENCIES DATAFRAME**')
st.write('')
st.write('')
st.write('')
d2=df2[['image','name','symbol','price_change_percentage_24h','current_price']]
d2['current_price']=df2['current_price'].round(decimals=1)


d2.drop([22,26,42,45,60,66,95],axis=0,inplace=True)
def path_to_image_html(path):
    return '<img src="' + path +'" width="40">'

st.markdown(d2.to_html(escape=False,formatters=dict(image=path_to_image_html)),unsafe_allow_html=True)

df2.to_html("webpage.html",escape=False,formatters=dict(image=path_to_image_html))


