import streamlit as st
import langchain_helper

st.title('Restaurant Name Generator')

cuisine = st.sidebar.selectbox('Pick a Cuisine', ('Indian', 'Italian', 'Bulgarian', 'Mexican', 'American'))

if cuisine:
    response = langchain_helper.generate_restaurant_name_and_items(cuisine)
    st.header(response['restaurant_name'].strip())
    menu_items = response['menu_items'].split('\n\n')
    st.write('**Menu Items**')
    for item in menu_items[1].split('\n'):
        st.write('-', item.split('.')[1])
        
