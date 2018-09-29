import numpy as np
import math
import random
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.metrics.pairwise import pairwise_distances

m=943
n=1682
result_matrix=np.zeros((m,n))
avg_ratings_user={}
avg_ratings_item={}
input_list=[]

#Average score of all the ratings given by a user
def populate_avg_ratings_user(train_matrix):
    for i in range(m):
        sum=0
        freq=0
        for j in range(n):
            if(train_matrix[i,j]!=0):
                freq+=1
                sum+=train_matrix[i,j]
        res=0
        if(freq!=0):
            res=sum*1.0/freq
        avg_ratings_user[i]=res

#Average score of an item based on user ratings
def populate_avg_ratings_item(train_matrix):
    for j in range(n):
        sum=0
        freq=0
        for i in range(m):
            if(train_matrix[i,j]!=0):
                freq+=1
                sum+=train_matrix[i,j]
        res=0
        if(freq!=0):
            res=sum*1.0/freq
        avg_ratings_item[j]=res


def predict(matrix,similarity_matrix,i,j):

    similarity_list=similarity_matrix[i]
    size=len(similarity_list)
    denom_sum=0
    numerator_sum=0
    for k in range(size):
        if(k!=i and matrix[k,j]!=0):
            flag=1
            denom_sum+=(1-similarity_list[k])
            numerator_sum+=((matrix[k,j]-avg_ratings_user[k])*(1-similarity_list[k]))

    result=avg_ratings_user[i]
    
    if(denom_sum!=0):
        result+=(numerator_sum*1.0/denom_sum)
    
    return result


def split_data(input_list, train_factor):
    total_size=len(input_list)
    matrix=np.zeros((m,n))
    train_matrix=np.zeros((m,n))
    test_matrix=np.zeros((m,n))
    train_size=math.floor(total_size*train_factor)
    test_size=total_size-train_size
    flag=0  #will switch between train_size and test_size
    for inp in input_list:
        i=int(inp[0])
        j=int(inp[1])
        val=int(inp[2])
        matrix[i-1,j-1]=val
        if(flag==0):
            if(train_size>0):
                train_matrix[i-1,j-1]=val
                train_size-=1
            else:
                test_matrix[i-1,j-1]=val
                test_size-=1
        else:
            if(test_size>0):
                test_matrix[i-1,j-1]=val
                test_size-=1
            else:
                train_matrix[i-1,j-1]=val
                train_size-=1

    flag=(flag+1)%2

    return matrix,train_matrix,test_matrix


def round_off(num):
    if(math.ceil(num)-num<= num-math.floor(num)):
        return int(math.ceil(num))
    else:
        return int(math.floor(num))


def calculate_error(error_vector):
    sum=0
    for x in error_vector:
        sum+=(x*x)
    
    return math.sqrt((sum*1.0)/len(error_vector))


def main():

    print("Loading Input data")
    with open("train_all_txt.txt") as f:
        for line in f:
            list=line.split()
            input_list.append(list)



    #Split data into train_matrix and test_matrix( specify train_matrix ratio as second parameter
    matrix,train_matrix,test_matrix=split_data(input_list, 0.90)

    populate_avg_ratings_user(train_matrix)
    populate_avg_ratings_item(train_matrix)


    #pairwise_distance calculates (1-similarity) between two users
    similarity_matrix = pairwise_distances(train_matrix, metric='cosine')

    fp=open('output.txt','w')

    print("Predicting rest of the values")


    #Use the data stored in train_matrix to predict values not filled in the main matrix (using user-based recommender).
    for i in range(m):
        for j in range(n):
            if(train_matrix[i,j]!=0):
                result_matrix[i,j]=train_matrix[i,j]
            else:
                predicted_value=round_off(predict(train_matrix,similarity_matrix,i,j))
                if(predicted_value==0):
                    if(avg_ratings_item[j]!=0):
                        predicted_value=avg_ratings_item[j]
                    else:
                        predicted_value=random.randint(2,4)
                        #On a scale of 1-5, if an item is not at all rated, give it an average rating: 3
                result_matrix[i,j]=predicted_value
            print(((i+1,j+1),int(result_matrix[i,j])))
            fp.write(str(i+1)+" "+str(j+1)+" "+str(int(result_matrix[i,j])))
            fp.write('\n')


    #Calculate error score based on difference in output in predicted values and test_matrix values
    error_vector=[]
    for i in range(m):
        for j in range(n):
            if(test_matrix[i,j]!=0):
                error_vector.append(test_matrix[i,j]-result_matrix[i,j])

    error_val=calculate_error(error_vector)
    print("Values predicted with error value: "+ str(error_val))

main()

