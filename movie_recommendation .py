#!/usr/bin/env python
# coding: utf-8

# # load the dataset




import graphlab

dataset = graphlab.SFrame.read_csv('C:\\Users\\Vidya sagar\\Documents\\ml-20m\\ml-20m1\\ratings.csv')

dataset.head()

items = graphlab.SFrame.read_csv('C:\\Users\\Vidya sagar\\Documents\\ml-20m\\ml-20m1\\movies.csv')


items.head()



items




items['title'].sketch_summary()



items.unique()



items['movieId'].show()


items['title'].show()




items['movieId=53835'].show()

mostly_liked_movie = items[items['movieId']==53835]
mostly_liked_movie.show()

training_data,test_data = dataset.random_split(.8,seed=0)

model=graphlab.recommender.create(training_data,'userId','movieId')





results = model.recommend()







