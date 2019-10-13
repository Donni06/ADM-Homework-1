#####################
# HackerRank Part 1 #
#####################

#################
# 1. INTRODUCTION

# 1.1 Solve me first
def solveMeFirst(a,b):
	# Hint: Type return a+b below
    return a+b;

num1 = int(input())
num2 = int(input())
res = solveMeFirst(num1,num2)
print(res)

# 1.2 Python If-else
import math
import os
import random
import re
import sys
if __name__ == '__main__':
  
    n = int(input().strip())  
    if n % 2 == 1:
        print("Weird")
    else:
        if n >= 2 and n <= 5:
            print("Not Weird")
        elif n % 2 == 0 and 6 <= n <= 20:
            print("Weird")
        elif n >= 20:
            print("Not Weird")

# 1.3 Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    c = a+b
    print(c)
    d = a-b
    print(d)
    e = a*b
    print (e)

# 1.4 Divisions
if __name__ == '__main__':
a = int(input())
b = int(input())
c = a//b
print(c)
d = a/b
print(d)

# 1.5 Loops
if __name__ == '__main__':
    n = int(input())
    i = 0
    while n>i:
        print(i*i)
        i+=1 

# 1.6 Say "Hello World"
if __name__ == '__main__':
    print ("Hello, World!")

# 1.7 Write a function
def is_leap(year):
    leap = False
    return year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)

# 1.8 Print a function
from __future__ import print_function
if __name__ == '__main__':
    n = int(raw_input())
    i=1
    while i<=n:
     print(i, end='')
     i+=1


############
# 2. BASIC DATA TYPES

# 2.1 Find the runner up
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int,input().strip().split())) 
    first = max(arr)
    for i in range (0,n):
        if max(arr) == first:
            arr.remove(max(arr))
    arr.sort()
    print (arr[-1])

# 2.2 Finding the percentage
if __name__ == '__main__':
    #I used the discussion section on HackerRank for this exercise
    n = int(input())
    student_marks = {}
    for i in range(0, n):
        name, *line = input().split()
        scores = list(map(float, line))
        scores=sum(scores)/3
        student_marks[name] = scores  
    query_name = input()    
    print('%.2f' % student_marks[query_name]) # print(output) formatted to 2 decimal places

# 2.3 Tuples
if __name__ == '__main__':
    n = int(input())
    integer_tuple = tuple(map(int, input().split()))
    print (hash(integer_tuple))

# 2.4 List comprehension
if __name__ == '__main__':   
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    arr = [[i, j, k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if  i + j + k != n] #nested for to access every elements of the three dimension matrix
    print(arr) 

# 2.5 Lists
l = []
N = int(input())
for i in range(0, N):
    command = input().split()
    if command[0] == 'insert':
        l.insert(int(command[1]), int(command[2]))
    elif command[0] == 'print':
        print (l)
    elif command[0] == 'remove':
        l.remove(int(command[1]))
    elif command[0] == 'append':
        l.append(int(command[1]))
    elif command[0] == 'sort':
        l.sort()
    elif command[0] == 'pop':
        l.pop()
    elif command[0] == 'reverse':
        l.reverse()

# 2.6 Nested lists
n = int(input())
students = []
for i in range(n):
    students.append([input(), float(input())])
scores = set([students[x][1] for x in range(n)]) #I created a set with every possible grade 
scores = list(scores) #change it to a list so we can sort it
scores.sort()

beststudents = [x[0] for x in students if x[1] == scores[1]] #created a list with every name/names x[0] associated with the highest mark
beststudents.sort()

for student in beststudents:
    print (student)


############
# 3. STRINGS

# 3.1 String Split and Join 
def split_and_join(line):
    a = line.split(" ")
    a = "-".join(a)

    return a

# 3.2 What's your name 
def print_full_name(a, b):
    c = a + " " + b+ "!"
    print("Hello", c, "You just delved into python." )

# 3.3 Mutations 
def mutate_string(string, position, character):
    l = string[:position] + character + string[position+1:]
    return l

# 3.4 Text allignment  
t = int(input()) #This must be an odd number
c = 'H'
#Top Cone
for i in range(t):
    print((c*i).rjust(t-1)+c+(c*i).ljust(t-1))
#Top Pillars
for i in range(t+1):
    print((c*t).center(t*2)+(c*t).center(t*6))
#Middle Belt
for i in range((t+1)//2):
    print((c*t*5).center(t*6))    
#Bottom Pillars
for i in range(t+1):
    print((c*t).center(t*2)+(c*t).center(t*6))    
#Bottom Cone
for i in range(t):
    print(((c*(t-i-1)).rjust(t)+c+(c*(t-i-1)).ljust(t)).rjust(t*6))

# 3.5 Text Wrap
def wrap(string, max_width):  
    return textwrap.fill (string, width=max_width)

# 3.6 Swap Case
def swap_case(s):
    return s.swapcase()

# 3.7 String formatting
# I got some help from the solutions
def print_formatted(number):
    width = len(bin(number).replace('0b',''))
    for i in range(1, number+1):
        print "{0:{width}d} {0:{width}o} {0:{width}X} {0:{width}b}".format(i,width = width)

# 3.8 String Validators
def check(s, f):
  for c in s:
    if f(c): return True  
  return False
if __name__ == '__main__':
  s = raw_input()
  print check(s, str.isalnum)
  print check(s, str.isalpha)
  print check(s, str.isdigit)
  print check(s, str.islower)
  print check(s, str.isupper)

# 3.9 Aplhabet rangoli
#Got some help from the discussion
import string
def print_rangoli(size): 
    alpha = string.ascii_lowercase
    L = []
    for i in range(size):
        s = "-".join(alpha[i:size])
        L.append((s[::-1]+s[1:]).center(4*n-3, "-"))
    print('\n'.join(L[:0:-1]+L))

# 3.10 Find a string
def count_substring(string, sub_string):  
    results = 0
    for i in range (len(string)):
        if string[i:i+len(sub_string)] == sub_string:
            results += 1
    return results

# 3.11 Capitalize

import string
def solve(s):
    s1 = s.split(' ')
    s2 = []
    for i in s1:
       s2.append(i.capitalize())
    s2 = " ".join(s2)    
    return s2

# 3.12 Merge the tools
from collections import OrderedDict
def merge_the_tools(string, k):
    lst = list(string)
    new = [lst[i:i+k] for i in range(0,len(lst),k)] 
    for i in new:
        print (*list(OrderedDict.fromkeys(i)), sep = '')

# 3.13 Designer Dot 
n, m = map(int, input().split())
for i in range(1, n, 2):
    print (str('.|.' * i).center(m, '-'))
print ('WELCOME'.center(m, '-'))
for i in range(n-2, -1, -2):
    print (str('.|.' * i).center(m, '-'))


#############
# 4. SETS

# 4.1 Introduction to sets
def average(array):
    array = set(array)
    av = sum(array)/len(array)
    return round(av, 3)

# 4.2 Set.add
n = int(input().strip())
arr = set([input().strip() for i in range(n)])
print len(arr)

# 4.3 Symmetric difference 
m = int(input())
set1 = set(map(int,input().split()))
n = int(input())
set2 = set(map(int,input().split()))
set3 = (set1.difference(set2)).union(set2.difference(set1))
for i in sorted(list(set3)):
        print (i)

# 4.4 set.union
n = int(input())
s1 = set(map(int, input().split()))
m = int(input())
s2 = set(map(int, input().split()))
print (len(set(s1.union(s2))))

# 4.5 set.intersection
n = int(input())
s1 = set(map(int, input().split()))
m = int(input())
s2 = set(map(int, input().split()))
print (len(set(s1.intersection(s2))))

# 4.6 set.difference
m = int(input())
s1 = set(map(int, input().split()))
n = int(input())
s2 = set(map(int, input().split()))
print (len(set(s1-s2)))

# 4.7 set.symmetric_difference
m = int(input())
s1 = set(map(int, input().split()))
n = int(input())
s2 = set(map(int, input().split()))
print (len(set(s1.symmetric_difference(s2))))

# 4.8 Set mutations
m = int(input())
s1 = set(map(int, input().split()))
n = int(input())
for i in range(n):
    (operation, other_set_length) = input().split()
    other_set = set(map(int, input().split()))
    if operation == 'intersection_update':
        s1.intersection_update(other_set)
    elif operation == 'update':
        s1.update(other_set)
    elif operation == 'symmetric_difference_update':
        s1.symmetric_difference_update(other_set)
    elif operation == 'difference_update':
        s1.difference_update(other_set)
    else:
        return False
print (sum(s1))

# 4.9 Set .discard(), .remove() & .pop() 
n = int(input())
s = set(map(int, input().split()))
m = int(input())
for _ in range(m):
    x = input().split()
    if x[0] == 'pop': # x[0] is the position that contains the name of the operation
        s.pop()
    elif x[0] == 'remove':
        try:
            s.remove(int(x[1])) # x[1] is the position that contains the values
        except:
            next
    elif x[0] == 'discard':
        s.discard(int(x[1]))
print (sum(s))

# 4.10 Check Subset
n = int(input())
for i in range(n):
    a, A = input(), set(map(int, input().split()))
    b, B = input(), set(map(int,input().split()))
    print(A.issubset(B))

# 4.11 The captain's room
n = int(input())
l1 = list(map(int, input().split()))
s1 = set(l1)
sum_room_list = sum(l1)   #in the example the sum is 31
sum_room_set = sum(s1)    #in our example the sum is 6
diff = sum_room_set * n - sum_room_list
print(diff // (n - 1)) # // is used for integer divisions

# 4.12 Check strict subset
a = set(map(int, input().split()))
n = int(input())
set_list= []
for i in range (n):
    set_list.append(set(map(int, input().split())))
count = 0
for i in (set_list):
    if i.issubset(a):
        count += 1
print (len(set_list) == count ) # Prints TRUE OR FALSE wether it's equal or not

# 4.13 No Idea
n, m = map(int, input().split())
l = map(int, input().split())
a = set(map(int, input().split()))
b = set(map(int, input().split()))
count = 0
for i in l:
    if i in a:
        count += 1
    elif i in b:
        count -=1
print(count)


#####################
# 5. Collections

# 5.1 Collections.deque()
def person_lister(f):
    def inner(people):
        return map(f,sorted(people, key=lambda x: int(x[2])))
    return inner

# 5.2 collections.Counter()
from collections import Counter
n = int(input())
shoes = (map(int, input().split()))
store = Counter(shoes)
c = int(input())
money = 0
for i in range (c):
    (size, price) = map(int, input().split())
    if store [size] > 0:
        store [size] -= 1
        money += price 
print (money)

# 5.3 collections.OrderDict()
from collections import OrderedDict
n= int(input())
d= OrderedDict() #I created and orderedDict istance
for i in range(n):
    item = input().split()
    itemPrice= int(item[-1])
    itemName= " ".join(item[:-1])
    if d.get(itemName): # .get is used to check if itemName already exists
       d[itemName] += itemPrice
    else:
       d[itemName] = itemPrice
for i in d.keys():
    print (i, d[i])

# 5.4 collections.namedTuples()
import collections
N = int(input())
columns = ','.join(input().split())
Student = collections.namedtuple('Student', columns)
sum = 0
for i in range(N):
    line = input().split()
    student = Student(*line)
    sum += int(student.MARKS)
print (sum / N)

# 5.5 DefaulDict Tutorial
from collections import defaultdict
d = defaultdict(list)
n, m = map(int, input().split())
for i in range(n):
    d[input()].append(i + 1)
for i in range(m):
    print(*d.get(input(), [-1]), sep=' ')

# 5.6 Piling up
ntest = int(input())
for _ in range (ntest):
    j = int(input())
    cube = list(map(int, input().split()))
    pile = 0
    if cube [0] > cube [-1]:
        last = cube [0]
    else:
        last = cube[-1]
    for i in range(0, len(cube)):
        if cube[0] <= last and cube [0] >= cube [-1]:
            last = cube.pop(0)
            pile += 1
        elif cube [-1] <= last and cube[-1] > cube [0]:
            last = cube.pop(-1)
            pile +=1
        else:
            print ("No")
            break
    if pile == j:
        print ("Yes")

# 5.7 Word order
from collections import Counter 
n = int(input())
words = [input() for i in range (n)]
s = len(set(words))
print (s)
print (*list(Counter(words).values()))

# 5.8 Company logo
from collections import Counter
s = str(input())
s = ''.join(sorted(s))
words = Counter(s).most_common(3)
for i in words:
    print (*i)


###################
# 6. DATE AND TIME 

# 6.1 Calendar module
import calendar
m, d, y = map(int, input().split())
print (calendar.day_name[calendar.weekday(y, m, d)].upper())

# 6.2 Time delta
#Got some help from the solutions 
from datetime import datetime
def time_delta(t1, t2):
    time_format = '%a %d %b %Y %H:%M:%S %z'
    t1 = datetime.strptime(t1, time_format) #get time from t1
    t2 = datetime.strptime(t2, time_format) #get time from t2
    return (int(abs((t1-t2).total_seconds()))) #we calculate the difference between the two and return the absolute value
t = int(input())
for t_itr in range(t):
    t1 = input()
    t2 = input()
    delta = time_delta(t1, t2)
    print(delta)


####################
# 7. EXCEPTIONS

# 7.1 Exceptions
# Complete the solve function below.
import string
def solve(s):
    s1 = s.split(' ')
    s2 = []
    for i in s1:
       s2.append(i.capitalize())
    s2 = " ".join(s2)    
    return s2


##################
# 8. BUILT INS

# 8.1 Any and alls
n = int(input())
l = list(map(int, input().split()))
if all (i > 0 for i in l) == True:
    if any([str(i) == str(i)[::-1] for i in l]): # if any element of string is equal to the last than print True
        print (True)
    else:
        print (False)
else:
    print (False)

# 8.2 Python evaluation
eval(input())

# 8.3 Input()
x,k = map(int, input().split())
print (k==eval(input())) 


#######################
# 9. PYTHON FUNCTIONALS

# 9.1 Reduce function
def product(fracs):
    t = reduce(lambda numerator, denominator : numerator * denominator, fracs)
    return t.numerator, t.denominator


#############################
# 10. REGEX AND PARSING

# 10.1 Re.split()
regex_pattern = r"[.,]+"

# 10.2 re.start() & re.end()
import re
S, k = input(), input()
mat = re.finditer(r'(?=(' + k + '))', S)
a = False
for match in mat:
    a = True
    print ((match.start(1), match.end(1) - 1))
if a == False:
    print ((-1, -1))


########################
# 11. XML

# 11.1 XML 1- Find the score
def get_attr_number(node):
    n=0
    n +=  len(node.attrib)
    for i in node:
        n += get_attr_number(i)
    return n
    

##############################
# 12. CLOSURES AND DECORATIONS

# 12.1 Standardize mobile number using decorators
def wrapper(f):
    def fun(l):
        fun = f("+91 "+c[-10:-5]+" "+c[-5:] for c in l) #takes values from every element of the list , and every list is divided as the exercise required
    return fun

# 12.2 Decorators 2 - Name directory
def person_lister(f):
    def inner(people):
        return map(f,sorted(people, key=lambda x: int(x[2])))
    return inner


#########################
# 13. NUMPY


# 13.1 Arrays
def arrays(arr):
    arr1 = numpy.array(arr, float)
    return numpy.flip(arr1)

# 13.2 Shape and reshape
import numpy as np
a = np.array(list(map(int,input().split())))
areshape = np.reshape(a, (3,3))
print(areshape)

# 13.3 Concatenate 
import numpy as np 
n, m, p = map(int,input().split())
arrn = np.array([input().split() for _ in range(n)],int)
arrm = np.array([input().split() for _ in range(m)],int)
print(np.concatenate((arrn, arrm), axis = 0))

# 13.4 Transpose and Flatten
import numpy as np
n, m= map(int, input().split())
array = np.array([input().strip().split() for i in range(n)], dtype = int)
print (np.transpose (array))
print (array.flatten())

# 13.5 Max and min
import numpy as np 
n, m = map(int, input().split())
a = np.array([input().strip().split() for i in range (n)], int)
print (np.max(np.min(a, axis = 1)))

# 13.6 Sum and prod
import numpy as np 
n, m = map(int, input().split())
a = np.array([input().strip().split() for i in range (n)], dtype = int)
print (np.prod(np.sum(a, axis = 0)))

# 13.7 Inner and Outer
import numpy as np
a = np.array(input().strip().split() , int)
b = np.array(input().strip().split() , int)
print (np.inner(a, b))
print (np.outer(a,b))

# 13.8 Zeros and ones
import numpy as np 
nums = tuple(map (int, input().split()))
print (np.zeros(nums, dtype=np.int))
print (np.ones(nums, dtype=np.int))

# 13.9 Dot and cross
import numpy as np
n = int(input())
a = np.array([input().split() for _ in range(n)], int)
b = np.array([input().split() for _ in range(n)], int)
print (np.dot(a,b))

# 13.10 Eye and identity
import numpy
numpy.set_printoptions(legacy='1.13')
n, m = map(int, input().split())
print (numpy.eye(n, m, k = 0)) 

# 13.11 Array Mathematics
import numpy 
N, M = list(map(int, input().split()))
a = numpy.array([input().split() for _ in range(N)], int)
b = numpy.array([input().split() for _ in range(N)], int)
print (numpy.add(a, b))         
print (numpy.subtract(a, b))     
print (numpy.multiply(a, b))     
print (a // b)       
print (numpy.mod(a, b))          
print (numpy.power(a, b))         

# 13.12 Floor, ceil and rint
import numpy as np
np.set_printoptions(sign=' ')
a = np.array(input().split(), dtype = float)
print (np.floor(a), np.ceil(a), np.rint(a), sep = '\n')

# 13.13 Mean, Var, and STD
import numpy as np
n, m = map(int, input().split())
a = np.array([input().split() for _ in range (n)], dtype = float)
np.set_printoptions(legacy = '1.13')
print (np.mean(a, axis = 1), np.var(a, axis = 0), np.std(a), sep = '\n' )

# 13.14 Linear Algebra
import numpy as np
n = int(input())
a = np.array([input().split() for _ in range (n)], dtype = float)
print (np.around(np.linalg.det(a), decimals = 2))

# 13.15 Polynomials
import numpy as  np
p = list(map(float,input().split()))
x = int(input())
print(np.polyval(p,int(x)))


#####################
# HackerRank Part 2 #
#####################


# Insertion 2
def insertionSort(arr):
    for i in range(1, len(arr)):
        j = i
        t = arr[i]
        while j > 0 and t < arr[j-1]:
            arr[j] = arr[j-1]
            j -= 1
        arr[j] = t
        print (' '.join(str(j) for j in arr))
n = input()
arr = list(map(int, input().split()))
insertionSort(arr)

# Birthday cake candles
import math
import os
import random
import re
import sys

def birthdayCakeCandles(ar):
    return ar.count(max(ar))    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    ar_count = int(input())
    ar = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(ar)
    fptr.write(str(result) + '\n')
    fptr.close()

# Viral advertisment 
import math
import os
import random
import re
import sys

def viralAdvertising(n):  
    people = 5
    count = 0
    for i in range (1,n+1):
        people = math.floor(people/2)
        count += people
        people = people*3
    return count

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()

# Kangaroo
import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    for n in range(10000): #10000 is the constraint written in the ex.
        if((x1+v1)==(x2+v2)):
            return "YES"
        x1+=v1
        x2+=v2
    return "NO"

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    x1V1X2V2 = input().split()
    x1 = int(x1V1X2V2[0])
    v1 = int(x1V1X2V2[1])
    x2 = int(x1V1X2V2[2])
    v2 = int(x1V1X2V2[3])
    result = kangaroo(x1, v1, x2, v2)
    fptr.write(result + '\n')
    fptr.close()

# Recursive digit sum WRONG APPROACH
def superDigit(n, k):
    p = n*k
    p = sum(map(int,str(p)))
    i = 1  
    while i > 0:
        if p > 9:
            p = sum(map(int,str(p)))
        else:
            return p     
n, k = input().split()
k = int (k)
print (superDigit(n, k))

# Recursive digit sum RIGHT APPROACH
def superDigit(n, k):
    n = int(n)
    k = int(k)
    p = n * k % 9
    return (p if p else 9)
       
n, k = input().split()
k = int (k)
print (superDigit(n, k))

# Insertion sort 1
import math
import os
import random
import re
import sys

# Complete the insertionSort1 function below.
def insertionSort1(n, arr):
    last = arr.pop()
    for i in range (len(arr)-1,0-2,  -1):
        if last >= arr[i] or i == -1:
            arr.insert(i+1, last)
            break
        else:
            arr.insert(i, arr[i])
            print (*arr, sep=" ")
            arr.pop(i)
    print (*arr, sep=" ")

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)





