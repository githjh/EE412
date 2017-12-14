###################################
#              EE412              #
#            Project #2           #
#     Collaborative Filtering     #
###################################

Name (Student ID)
Inho Cho (20154553)
Byeoksan Lee (20163439)
Juhyeng Han (20163683)

1. Method
  We use the neighborhood method to predict the user's rating on the item. Based on the raw rating matrix R, we calculate the D_item which represents cosine similarity between two different items and the D_user which represents cosine similarity between two different users. Based on the cosine similarity, we compute the weighted sum of the ten most relavent ratings. We selected the ten most relevant (with highest absolute values) ratings from similar items and another ten most relevant ratings from similar users. 
  For the tiemstamp the rating was recorded to be taken into account, we normalize the timestamps so that all timestamp value to be transformed to the value between 0 and 1. 0 means the oldest unix timestamp and 1 is the most recent timestamp (typically set to current time). And for each rating we gave the weights exponentially decreasing over time so that more recent data affect the predicted rating more. 


2. Implementation
  We adopt object-oriented approach for this project. We defined the 'RatingPredictor' class and according to the command line input, RatingPredictor performs its tasks. We tried to optimize the operation using matrix operation as much as possible using numpy and scipy Python package. For the cosine similarity, we utilized an efficient algorithm from sklearn package. 
  For the training, train() function is implemented in 'RatingPredictor' and validate() for validating the dataset, evaluate() for evaluating test dataset and predict() for predicting the estimated output from out algorithm. 


3. Required Package
  As we implemented RatingPredictor using Python, some Python packages shoulb be installed beforehand. Required Python packages are:
        scipy, numpy, sklearn
  You can easily install those packages using pip. i.e.

      $ sudo pip install scipy numpy sklearn


4. How to run
  To train the dataset,

     $ python rating.py train TRAIN_DATA

  Note that TRAIN_DATA should be tab-separated file with USER_ID, ITEM_ID, RATING, and UNIX TIMESTAMP at each column. Once you train the model, the model file 'model.pyz' will be created.
  To validate the dataset,

     $ python rating.py validate VALIDATE_DATA

  You should train the data first before validate data, validata data format should be the same as the training data. i.e. tab-separated file with USER_ID, ITEM_ID, RATING, and UNIX TIMESTAMP. This will calculate the RMSE for the validation dataset.
  To evaluate the dataset,

     $ python rating.py evaluate TEST_DATA OUTPUT_FILE

  TEST_DATA shuld be comma separated foramt (csv) with USER_ID, ITEM_ID, and UNIX TIMESTAMP. The predicted rating will be written on OUTPUT_FILE with the same order as the TEST_DATA with csv format.

5. References
[1] EE412 Lecture Note, "Lecture 14: Collaborative Filtering", KAIST KLMS
