Movie Recommending System

Shubham Balasaheb Patil
Python Developer Intern

1. INTRODUCTION
A recommendation system is a type of information filtering system which attempts to predict the preferences of a user, and make suggests based on these preferences.
There are a wide variety of applications for recommendation systems. These have become increasingly popular over the last few years and are now utilized in most online platforms that we use. The content of such platforms varies from movies, music, books and videos, to friends and stories on social media platforms, to products on e-commerce websites, to people on professional and dating websites, to search results returned on Google.

2. CONTENT BASED RECOMMENDATION
Content Based Recommendation algorithm considers the likes and dislikes of the user and generates a User Profile. For generating a user profile, we consider the item profiles (vector describing an item) and their corresponding user rating. The user profile is the weighted sum of the item profiles with weights being the ratings user rated. Once the user profile is generated, we calculate the similarity of the user profile with all the items in the dataset, which is calculated using cosine similarity between the user profile and item profile.

3. IMPLEMENTATION
All users exhibit varying rating behaviours. Some users are lenient in their ratings, whereas some are very stringent giving lower ratings to almost all movies. This user bias needs to be incorporated into the rating predictions. We compute the average rating for each user. The average rating of the user is then used as the prediction for each missing rating entry for that particular user. This method can be expected to perform slightly better than the global average since it takes into account the rating behaviour of the users into account.

