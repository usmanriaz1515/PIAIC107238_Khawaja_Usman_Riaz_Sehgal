#!/usr/bin/env python
# coding: utf-8

# Q1. Import numpy as np and print the version number.

# In[1]:


import numpy as np
print(np.__version__)


# Q2. Create a 1D array of numbers from 0 to 9

# In[2]:


array = np.arange(10)
array


# Q3. Create a 3×3 numpy array of all True’s

# In[3]:


arrayb = np.full((3,3),True,dtype=bool) 
arrayb


# Q4. Extract all odd numbers from arr

# In[4]:


num = np.arange(10)
num[num % 2 !=0]


# Q5. Replace all odd numbers in arr with -1

# In[5]:


num = np.arange(10)
num[num % 2 !=0]=-1
num


# Q6. Replace all odd numbers in arr with -1 without changing arr

# In[6]:


num = np.arange(10)
odd = np.where(num % 2 != 0, -1, num)
print(num)
print(odd)


# Q7. Convert a 1D array to a 2D array with 2 rows

# In[7]:


num = np.arange(10)
num.reshape(2,5)


# Q8. Stack arrays a and b vertically

# In[8]:


a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)
np.concatenate([a, b], axis=0)


# Q9. Stack the arrays a and b horizontally.

# In[9]:


a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)
np.concatenate([a, b], axis=1)


# Q10. Create the following pattern without hardcoding. Use only numpy functions and the below input array a.

# In[10]:


a = np.array([1,2,3])
np.r_[np.repeat(a, 3), np.tile(a, 3)]


# Q11. Get the common items between a and b

# In[11]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a,b)


# Q12. From array a remove all items present in array b

# In[12]:


a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
np.setdiff1d(a,b)


# Q13. Get the positions where elements of a and b match

# In[13]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])

np.where(a == b)


# Q14. Get all items between 5 and 10 from a.

# In[14]:


a = np.arange(15)
index = np.where((a >= 5) & (a <= 10))
a[index]


# Q15. Convert the function maxx that works on two scalars, to work on two arrays.

# In[15]:


def maxx(x, y):
    if x >= y:
        return x
    else:
        return y

pair_max = np.vectorize(maxx, otypes=[float])

a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])

pair_max(a, b)


# Q16. Swap columns 1 and 2 in the array arr.

# In[16]:


arr = np.arange(9).reshape(3,3)
arr[:, [1,0,2]]


# Q17. Swap rows 1 and 2 in the array arr:

# In[17]:


arr = np.arange(9).reshape(3,3)
arr[[1,0,2], :]


# Q18. Reverse the rows of a 2D array arr.

# In[18]:


arr = np.arange(9).reshape(3,3)
arr[::-1]


# Q19. Reverse the columns of a 2D array arr

# In[19]:


arr = np.arange(9).reshape(3,3)
arr[:, ::-1]


# Q20. Create a 2D array of shape 5x3 to contain random decimal numbers between 5 and 10.

# In[20]:


arr = np.arange(9).reshape(3,3)
rand_arr = np.random.uniform(5,10, size=(5,3))
print(rand_arr)


# Q21. Print or show only 3 decimal places of the numpy array rand_arr.

# In[21]:


rand_arr = np.random.random((5,3))
np.set_printoptions(precision=3)
rand_arr[:4]


# Q22. Pretty print rand_arr by suppressing the scientific notation (like 1e10)

# In[22]:


np.set_printoptions(suppress=False)
np.random.seed(100)
rand_arr = np.random.random([3,3])/1e3
rand_arr


# Q23. Limit the number of items printed in python numpy array a to a maximum of 6 elements.

# In[23]:


np.set_printoptions(threshold=6)
a = np.arange(15)
a


# Q24. Print the full numpy array a without truncating.

# In[24]:


np.set_printoptions(threshold=6)
a = np.arange(15)
np.set_printoptions(threshold=15)
a


# Q25. Import the iris dataset keeping the text intact.

# In[25]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
iris[:3]


# Q26. Extract the text column species from the 1D iris imported in previous question.

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
# print(iris_1d.shape)Q. Find the mean, median, standard deviation of iris's sepallength (1st column)
# species = np.array([row[4] for row in iris_1d])
# species[:5]

# Q27. Convert the 1D iris to 2D array iris_2d by omitting the species text field.

# In[26]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
iris_2d = np.array([row.tolist()[:4] for row in iris_1d])
iris_2d[:4]


# Q28. Find the mean, median, standard deviation of iris's sepallength (1st column)
# 

# In[27]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
mu, med, sd = np.mean(sepallength), np.median(sepallength), np.std(sepallength)
print(mu, med, sd)


# Q29. Create a normalized form of iris's sepallength whose values range exactly between 0 and 1 
# so that the minimum has value 0 and maximum has value 1.

# In[28]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
Smax, Smin = sepallength.max(), sepallength.min()
S = (sepallength - Smin)/sepallength.ptp()
print(S)


# Q30. Compute the softmax score of sepallength.

# In[29]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
sepallength = np.array([float(row[0]) for row in iris])

def softmax(x):
    """Compute softmax values for each sets of scores in x.
    https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

print(softmax(sepallength))


# Q31. Find the 5th and 95th percentile of iris's sepallength

# In[30]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
np.percentile(sepallength, q=[5, 95])


# Q32. Insert np.nan values at 20 random positions in iris_2d dataset

# In[31]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')

i, j = np.where(iris_2d)
np.random.seed(100)
iris_2d[np.random.choice((i), 20), np.random.choice((j), 20)] = np.nan
print(iris_2d[:10])


# Q33. Find the number and position of missing values in iris_2d's sepallength (1st column)

# In[32]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
print("Number of missing values: \n", np.isnan(iris_2d[:, 0]).sum())
print("Position of missing values: \n", np.where(np.isnan(iris_2d[:, 0])))


# Q34. Filter the rows of iris_2d that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0
# 
# 

# In[33]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
condition = (iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)
iris_2d[condition]


# Q35. Select the rows of iris_2d that does not have any nan value.

# In[34]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
any_nan_in_row = np.array([~np.any(np.isnan(row)) for row in iris_2d])
iris_2d[any_nan_in_row][:5]


# Q36. Find the correlation between SepalLength(1st column) and PetalLength(3rd column) in iris_2d

# In[35]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
np.corrcoef(iris[:, 0], iris[:, 2])[0, 1]


# Q37. Find out if iris_2d has any missing values.

# In[36]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

np.isnan(iris_2d).any()


# Q38. Replace all ccurrences of nan with 0 in numpy array

# In[37]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
iris_2d[np.isnan(iris_2d)] = 0
iris_2d[:4]


# Q39. Find the unique values and the count of unique values in iris's species

# In[38]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

species = np.array([row.tolist()[4] for row in iris])

np.unique(species, return_counts=True)


# Q40. Bin the petal length (3rd) column of iris_2d to form a text array, such that if petal length is:

# In[39]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# Bin petallength 
petal_length_bin = np.digitize(iris[:, 2].astype('float'), [0, 3, 5, 10])

# Map it to respective category
label_map = {1: 'small', 2: 'medium', 3: 'large', 4: np.nan}
petal_length_cat = [label_map[x] for x in petal_length_bin]

# View
petal_length_cat[:4]


# Q41. Create a new column for volume in iris_2d, where volume is (pi x petallength x sepal_length^2)/3

# In[40]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')

# Solution
# Compute volume
sepallength = iris_2d[:, 0].astype('float')
petallength = iris_2d[:, 2].astype('float')
volume = (np.pi * petallength * (sepallength**2))/3

# Introduce new dimension to match iris_2d's
volume = volume[:, np.newaxis]

# Add the new column
out = np.hstack([iris_2d, volume])

# View
out[:4]


# Q42. Randomly sample iris's species such that setose is twice the number of versicolor and virginica

# In[41]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')

# Solution
# Get the species column
species = iris[:, 4]

np.random.seed(100)
a = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
species_out = np.random.choice(a, 150, p=[0.5, 0.25, 0.25])
print(np.unique(species_out, return_counts=True))


# Q43. What is the value of second longest petallength of species setosa

# In[42]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')

# Solution
# Get the species and petal length columns
petal_len_setosa = iris[iris[:, 4] == b'Iris-setosa', [2]].astype('float')

# Get the second last value
np.unique(np.sort(petal_len_setosa))[-2]


# Q44. Sort the iris dataset based on sepallength column.

# In[43]:


print(iris[iris[:,0].argsort()][:20])


# Q45. Find the most frequent value of petal length (3rd column) in iris dataset.

# In[44]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')

# Solution:
vals, counts = np.unique(iris[:, 2], return_counts=True)
print(vals[np.argmax(counts)])


# Q46. Find the position of the first occurrence of a value greater than 1.0 in petalwidth 4th column of iris dataset.

# In[45]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')

# Solution: (edit: changed argmax to argwhere. Thanks Rong!)
np.argwhere(iris[:, 3].astype(float) > 1.0)[0]


# Q47. From the array a, replace all values greater than 30 to 30 and less than 10 to 10.

# In[46]:


np.set_printoptions(precision=2)
np.random.seed(100)
a = np.random.uniform(1,50, 20)

# Solution 1: Using np.clip
np.clip(a, a_min=10, a_max=30)


# Q48. Get the positions of top 5 maximum values in a given array a.
# 
# 

# In[47]:


np.random.seed(100)
a = np.random.uniform(1,50, 20)

# Solution:
print(a.argsort())


# Q49. Compute the counts of unique values row-wise.

# In[48]:


np.random.seed(100)
arr = np.random.randint(1,11,size=(6, 10))
arr


# Q50. Convert array_of_arrays into a flat linear 1d array.

# In[49]:


arr1 = np.arange(3)
arr2 = np.arange(3,7)
arr3 = np.arange(7,10)

array_of_arrays = np.array([arr1, arr2, arr3])
print('array_of_arrays: ', array_of_arrays)

# Solution 1
arr_2d = np.array([a for arr in array_of_arrays for a in arr])
print(arr_2d)

