import streamlit as st
import pickle
import numpy as np
import gzip
# Load the model and label encoders
with gzip.open('restaurant_rating_model.pkl.gz', 'rb') as file:
    data = pickle.load(file)
    model = data['model']
    label_encoders = data['label_encoders']

st.title("Restaurant Rating Predictor")

# Predefined lists for dropdowns
location_list = ['Banashankari', 'Basavanagudi', 'Jayanagar', 'Kumaraswamy Layout',
       'Rajarajeshwari Nagar', 'Vijay Nagar', 'Mysore Road',
       'Uttarahalli', 'South Bangalore', 'Bannerghatta Road', 'JP Nagar',
       'BTM', 'Kanakapura Road', 'Wilson Garden', 'Shanti Nagar',
       'Koramangala 5th Block', 'Richmond Road', 'City Market',
       'Bellandur', 'Sarjapur Road', 'Marathahalli', 'HSR',
       'Old Airport Road', 'Indiranagar', 'Koramangala 1st Block',
       'East Bangalore', 'MG Road', 'Brigade Road', 'Lavelle Road',
       'Church Street', 'Ulsoor', 'Residency Road', 'Shivajinagar',
       'Infantry Road', 'St. Marks Road', 'Cunningham Road',
       'Race Course Road', 'Domlur', 'Koramangala 8th Block',
       'Frazer Town', 'Ejipura', 'Vasanth Nagar', 'Jeevan Bhima Nagar',
       'Old Madras Road', 'Commercial Street', 'Koramangala 6th Block',
       'Majestic', 'Langford Town', 'Koramangala 7th Block',
       'Brookefield', 'Whitefield', 'ITPL Main Road, Whitefield',
       'Varthur Main Road, Whitefield', 'Koramangala 3rd Block',
       'Koramangala 2nd Block', 'Koramangala 4th Block', 'Koramangala',
       'Bommanahalli', 'Hosur Road', 'Seshadripuram', 'Electronic City',
       'Banaswadi', 'North Bangalore', 'RT Nagar', 'Kammanahalli',
       'Hennur', 'Nagawara', 'HBR Layout', 'Kalyan Nagar', 'Thippasandra',
       'CV Raman Nagar', 'Kaggadasapura', 'Rammurthy Nagar', 'Kengeri',
       'Sankey Road', 'Central Bangalore', 'Malleshwaram',
       'Sadashiv Nagar', 'Basaveshwara Nagar', 'Rajajinagar',
       'New BEL Road', 'West Bangalore', 'Yeshwantpur', 'Sanjay Nagar',
       'Sahakara Nagar', 'Jalahalli', 'Hebbal', 'Yelahanka',
       'Magadi Road', 'KR Puram']  # Add all locations
rest_type_list = ['Casual Dining', 'Cafe, Casual Dining', 'Quick Bites',
       'Casual Dining, Cafe', 'Cafe', 'Quick Bites, Cafe',
       'Cafe, Quick Bites', 'Delivery', 'Dessert Parlor',
       'Bakery, Dessert Parlor', 'Pub', 'Fine Dining', 'Beverage Shop',
       'Bakery', 'Bar', 'Takeaway, Delivery', 'Sweet Shop',
       'Beverage Shop, Quick Bites', 'Beverage Shop, Dessert Parlor',
       'Pub, Casual Dining', 'Casual Dining, Bar',
       'Dessert Parlor, Beverage Shop', 'Microbrewery, Casual Dining',
       'Sweet Shop, Quick Bites', 'Lounge', 'Food Truck', 'Cafe, Bakery',
       'Food Court', 'Microbrewery', 'Quick Bites, Dessert Parlor',
       'Kiosk', 'Pub, Bar', 'Casual Dining, Pub', 'Lounge, Bar',
       'Bakery, Quick Bites', 'Bar, Casual Dining',
       'Casual Dining, Microbrewery', 'Mess', 'Cafe, Dessert Parlor',
       'Dessert Parlor, Cafe', 'Quick Bites, Sweet Shop', 'Takeaway',
       'Microbrewery, Pub', 'Club', 'Bakery, Cafe',
       'Dessert Parlor, Quick Bites', 'Beverage Shop, Cafe', 'Pub, Cafe',
       'Casual Dining, Irani Cafee', 'Food Court, Quick Bites',
       'Quick Bites, Beverage Shop', 'Fine Dining, Lounge',
       'Quick Bites, Bakery', 'Bar, Quick Bites', 'Pub, Microbrewery',
       'Microbrewery, Lounge', 'Fine Dining, Microbrewery',
       'Fine Dining, Bar', 'Quick Bites, Food Court', 'Cafe, Bar',
       'Lounge, Casual Dining', 'Casual Dining, Lounge',
       'Microbrewery, Bar', 'Cafe, Lounge', 'Bar, Pub', 'Lounge, Cafe',
       'Club, Casual Dining', 'Dhaba', 'Dessert Parlor, Bakery',
       'Lounge, Microbrewery', 'Bar, Lounge', 'Food Court, Casual Dining']  # Add all restaurant types
cuisines_list =['North Indian', 'Mughlai', 'Chinese', 'Thai', 'Cafe', 'Mexican',
       'Italian', 'South Indian', 'Rajasthani', 'Pizza', 'Continental',
       'Momos', 'Beverages', 'Fast Food', 'American', 'French',
       'European', 'Burger', 'Biryani', 'Street Food', 'Rolls',
       'Ice Cream', 'Desserts', 'Andhra', 'Healthy Food', 'Salad',
       'Asian', 'Korean', 'Indonesian', 'Japanese', 'Goan', 'Seafood',
       'Bakery', 'Kebab', 'Steak', 'Sandwich', 'Juices', 'Vietnamese',
       'Mithai', 'Hyderabadi', 'Arabian', 'BBQ', 'Mangalorean', 'Tea',
       'Afghani', 'Finger Food', 'Tibetan', 'Middle Eastern',
       'Mediterranean', 'Bengali', 'Kerala', 'Charcoal Chicken', 'Oriya',
       'Bihari', 'Roast Chicken', 'African', 'Lebanese', 'Belgian',
       'South American', 'Maharashtrian', 'Konkan', 'Chettinad', 'Wraps',
       'Turkish', 'Coffee', 'Afghan', 'Modern Indian', 'Iranian',
       'Lucknowi', 'Gujarati', 'Tex-Mex', 'Tamil', 'Spanish', 'Malaysian',
       'Burmese', 'Portuguese', 'Parsi', 'Nepalese', 'Greek',
       'North Eastern', 'Bar Food', 'Singaporean', 'Awadhi', 'Naga',
       'Cantonese', 'Sushi', 'Bubble Tea', 'Kashmiri', 'Assamese',
       'Sri Lankan', 'Grill', 'British', 'German', 'Russian', 'Bohri',
       'Jewish', 'Vegan', 'Sindhi'] # Add all cuisines
listed_type_list = ['Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out',
       'Drinks & nightlife', 'Pubs and bars']  # Add all listed types

# Creating input fields with dropdowns
delivery = st.selectbox('Delivery', ['Yes', 'No'])
booking = st.selectbox('Booking', ['Yes', 'No'])
votes = st.slider('Votes', min_value=50,max_value=16832,step=10)  # Example list for votes; adjust based on actual values
location = st.selectbox('Location', location_list)
rest_type = st.selectbox('Restaurant Type', rest_type_list)
cuisines = st.multiselect('Cuisines', cuisines_list)
cost_of_two_people = st.slider('Cost For 2 People', min_value=40,max_value=6000, step=10)
listed_type = st.selectbox('Listed Type', listed_type_list)

# When the user clicks the "Predict" button
if st.button('Predict'):
    input_data = {
        'Delivery': delivery,
        'Booking': booking,
        'Votes': int(votes),
        'Location': location,
        'Rest_Type': rest_type,
        'Cuisines': ', '.join(cuisines),  # Join selected cuisines as a single string
        'Cost_of_Two_People': int(cost_of_two_people),
        'Type': listed_type
    }

    # Encode the categorical variables using the stored label encoders
    for key in input_data.keys():
        if key in label_encoders:
            input_data[key] = label_encoders[key].transform([input_data[key]])[0]

    # Convert input data to numpy array
    features = np.array(list(input_data.values())).reshape(1, -1)

    # Predict the rating
    prediction = model.predict(features)[0]
    st.success(f'Predicted Rating: {prediction:.2f}')
