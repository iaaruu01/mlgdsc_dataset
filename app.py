import numpy as np
import pandas as pd
import pickle
import streamlit as st

csv_file_path = 'advertising.csv'
df = pd.read_csv(csv_file_path)

pickle_file_path = 'advertising.pkl'
df.to_pickle(pickle_file_path)
loaded_model = pickle.load(open('advertising.pkl','rb'))

def banknote_predict(input_data):
    # changing the array as we are predicting for one instance
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
        return 'Clicked on Ad'
    
    else:
        return 'Not Clicked on Ad'
st.title('Advertising on Ad')

  
st.divider()
with st.expander('Project Features'):
  st.write('*The Project takes input of given features :*')
  st.caption('1. Variance')
  st.caption('2. skewness')
  st.caption('3. Curtosis')
  st.caption('4. Entropy')


def main():
    st.subheader('Enter your values')
    variance = st.text_input('Enter Variance')
    skewness = st.text_input('Enter skewness')
    curtosis = st.text_input('Enter Curtosis')
    entropy = st.text_input('Enter Entropy')

    prediction = ''

    if st.button('Predict the result'):
        prediction = banknote_predict([variance,skewness,curtosis,entropy])

    st.success(prediction)
    st.divider()
    st.subheader('Made with 💌 by GDSC KIET')

if __name__ =='__main__':
     main()