# Count the Occurrence of Each Character in a String

str='Hello World'

def occur(str):
    arr_str= list(str)
    map={}
    for ele in arr_str:
        if ele == ' ':
            continue
        elif ele in map :
            map[ele]=map[ele]+1
        else :
            map[ele] = 1 ;
    return map
print(occur(str))

#  Remove Duplicate Elements from a List

lst=[1,23,3,3,3,4,4,5]

def rem_dub(arr):
    for i in range(len(arr)):
        for j in range(i+1,len(arr)):
            if arr[j] == arr[i] :
                arr[j] = 0
    return [x for x in arr if x!=0]
print(rem_dub(lst))

def prime_num(num):
    count =0
    for i in range(1,num+1):
        if num%i==0:
            count=count +1
    if count == 2 :
        return True
    else:
        return False
print(prime_num(7))
print(prime_num(10))

def pair(arr,sum):
    for i in range(0,len(arr)):
        for j in range(0,len(arr)):
            if (i == j) :
                continue
            elif (arr[i] + arr[j] == sum):
                return (i,j)
    
print('pair',pair([0, -1, 2, -3, 1],-2))

# Max Sum subarray

def Maxsum(arr):
    maxsum=arr[0]
    cursum = 0
    for num in arr:
        if cursum<0:
            cursum=0
        cursum = cursum + num
        maxsum = max(maxsum,cursum)
    return maxsum

print('maxsum',Maxsum([2, 3, -8, 7, -1, 2, 3]))