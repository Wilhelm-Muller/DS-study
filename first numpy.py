import numpy as np
import pandas as pd
# print(np.__version__)

# 1. Numpy array format => np.array()
#array = np.array([[1,2,3],[4,5,6]])
#print(array)
#print("number of dimension:",array.ndim)
#print("shape",array.shape)
#print("total size:", array.size)

# 2. Numpy array with different variations
#zeros = np.zeros((2,3))
#print(zeros)

#float = np.array([[1,2,3],[4,5,6]], dtype=float)
#print(float)
# dtype attribute is used to define the data type of the array

#emp = np.empty((2,3))
# empty function creates an array without initializing its values

#arrange = np.arange(12).reshape(4,3)
#print(arrange)
#reshape(no_of_rows, no_of_columns)

#a = np.linspace(0, 10, 5).reshape(1,5)
#print(a)
#linspace(start, stop, no_of_elements) function creates an array of evenly spaced numbers over a specified interval.


# 3. Basic operations on numpy array
#a = np.array([10,20,30,40])
#b= np.arange(4)
#print(a-b)

# dot product of two arrays = np.dot(a,b)
#print(np.dot(a,b))

# cross product of two arrays = np.cross(a,b)
#print(np.cross(a,b))

# Randomization of numpy array
#a = np.random.random((2,4))
#print(a)
#print(np.max(a))
#print(np.min(a))
#print(np.mean(a))
#print(np.sum(a))
#np.random.random() generates random numbers between 0 and 1, with arguments as shape of the array
#np.max() returns the maximum value of the array, and min, mean and sum returns the minimum, mean and sum of the array respectively
# axis = 0 means column wise operation and axis = 1 means row wise operation

#A = np.arange(2,17).reshape(3,5)
#print(np.argmin(A)) #position of minimum value in the array
#print(np.argmax(A)) #position of maximum value in the array
#print(np.median(A)) #median of the array
#print(np.cumsum(A)) #cumulative sum of the array
#print(np.diff(A)) #difference between consecutive elements
#print(np.nonzero(A)) #non zero elements of the array
#print(np.transpose(A)) #transpose of the array
#print(np.sort(A)) #sort the array in ascending order
#print(np.clip(A, 5, 10)) #clip the values of the array between 5 and 10
#print(A)

# 4.Indexing and slicing of numpy array
#A = np.arange(3,15).reshape(3,4)
#print(A)
#print(A[2][1]) #accessing element at 2nd row and 1st column
#print(A[1:3, 1:3]) #slicing the array from 1st to 3rd row and 1st to 3rd column
#print(A.flatten()) #flattening the array to 1D, A.flat is the iterator of the flattened array

#A=A.reshape(4,3)
#B= np.array([[1,2,3],[4,5,6]])
#c = np.vstack((A,B)) # vertical stacking of two arrays
#print(c.shape) #4+2 = 6 rows and 3 columns 
# d = np.hstack((A,B)) # horizontal stacking of two arrays
#print(A[np.newaxis, :]) #adding a new axis to the array, A[np.newaxis, :] is equivalent to A.reshape(1,4,3)

#A=np.array([1,1,1])[:, np.newaxis]
#B=np.array([2,2,2])[:, np.newaxis]
#C= np.concatenate((A,B,B,A), axis=0) #concatenating two arrays along the specified axis, 0->row wise and 1->column wise
#print(C)


# 5. Slicing of numpy array
#A = np.arange(12).reshape(3,4)
#print(A)

#print(np.split(A, 2, axis=1)) #splitting the array into 2 parts along the specified axis, 0->row wise and 1->column wise
#print(np.array_split(A, 3, axis=1)) #splitting the array into 3 parts along the specified axis, 0->row wise and 1->column wise
#print(np.hsplit(A, 2)) #splitting the array into 2 parts horizontally
#print(np.vsplit(A, 3)) #splitting the array into 3 parts vertically


# 6. Deep copy and shallow copy of numpy array
a = np.arange(4)
b = a # shallow copy of a
c = a.copy() # deep copy of a
print(a)
a[0] = 10
print(b)
print(c)
# shallow copy means that the new array is a view of the original array,
# so any changes made to the new array will also affect the original array. 
# Deep copy means that the new array is a copy of the original array, so any 
# changes made to the new array will not affect the original array.

