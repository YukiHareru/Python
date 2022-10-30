# Author: Shen Qiu Tong
# Student ID: 30763819
# Task 1
import math
import random

def counting_sort_stable_number_for_radixSort(lst, base, col = None):
    '''
    This function do counting sort for the radix sort.
    Find the largest elements in the array to be sorted;
    store the item in specific column according to
    the base given in the i-th item of the counting array;
    then sort the original array by the counting array
    :param lst: the list we need to sort
    :param base: the base given to sort the list
    :param col: the column of the number
    :return: a sorted list base on the column
    complexity: O(n+b) where n is the length of nums and
    b is the value of base
    Reference: Dr Lim Wern Han's lecture
    '''
    count_array = [None] * (base + 1)
    # create a count array that has base + 1 buckets
    # + 1 for 0
    for i in range(len(count_array)):
        count_array[i] = []
    for item in lst:
        count_array[item // (base ** col) % base].append(item)
    index = 0
    for i in range(len(count_array)):
        frequency = count_array[i]
        for j in range(len(frequency)):
            lst[index] = count_array[i][j]
            index = index + 1
    return lst

def num_rad_sort(nums, b):
    '''
    This function sorting the nums by using radix sort
    by the base given.
    :param nums: the list need to be sorted
    :param b: base used to sort the list
    :return: a sorted list
    complexity is O((n + b) ∗ logbM) where
    n is the length of nums, b is the value of
    base and M is the numerical value of the
    maximum element of nums
    Reference: Dr Lim Wern Han's lecture
    '''
    if nums == []:
        return []
    max_num = 0
    for i in nums:
        if i > max_num:
            max_num = i
    if max_num == 0:
        max_col = 1
    else:
        max_col = math.log(max_num,b) + 1
    # + 1 in case that base is same as the biggest number
    col = 0
    while col < max_col:
        counting_sort_stable_number_for_radixSort(nums, b, col)
        col += 1
    return nums
# nums = [43, 101, 22, 27, 5, 50, 15]
# print(num_rad_sort(nums, 4))

# Task 2
import time
def  base_timer(num_list, base_list):
    """
    This function calculate the time taken for sorting
    the input list using different bases in base_list
    :param num_list: the list need to be sorted
    :param base_list: the bases list used to sort the list
    :return: time taken for sorting the num_list by using the base in base_list
    complexity is O(k * (n + b) ∗ logbM) where
    n is the length of nums, b is the value of
    base, M is the numerical value of the
    maximum element of nums and k is the length of
    base_list
    """
    time_used = []
    # code you want to time
    for base in base_list:
        start = time.time()
        num_rad_sort(num_list, base)
        time_taken = time.time() - start
        time_used.append(time_taken)
    return time_used
# random.seed("FIT2004S22021")
# data1 = [random.randint(0,2**25) for _ in range(2**15)]
# data2 = [random.randint(0,2**25) for _ in range(2**16)]
# bases1 = [2**i for i in range(1,23)]
# bases2 = [2*10**6 + (5*10**5)*i for i in range(1,10)]
# y1 = base_timer(data1, bases1)
# print(y1)
# y2 = base_timer(data2, bases1)
# print(y2)
# y3 = base_timer(data1, bases2)
# print(y3)
# y4 = base_timer(data2, bases2)
# print(y4)
# import matplotlib.pyplot as plt
# base = []
# for i in bases1:
#     base.append(math.log(i))
# plt.plot(base, y1, label = "y1")
# plt.plot(base, y2, label = "y2")
# plt.xlabel('bases1')
# plt.ylabel('runtimes')
# plt.legend()
# plt.show()
# plt.plot(bases2, y3, label = "y3")
# plt.plot(bases2, y4, label = "y4")
# plt.xlabel('bases2')
# plt.ylabel('runtimes')
# plt.legend()
# plt.show()

# Task 3
def count_sort_letters(lst, size, col, base):
    '''
    Reference: https://stackoverflow.com/questions/60968950/radix-sort-for-strings-in-python
    Have do modification on the code
    Helper routine for performing a count sort based upon column col
    :param lst: the list of hobbies set need to be sorted
    :param size: the size of the input list
    :param col: the column of the stting
    :param base: the base used to sort the list
    :return: a sorted list by column
    complexity is O(27) because we always create 27 buckets for sorting
    the interest group
    '''
    output = [None] * size
    count = [None] * (base + 2)  # One addition cell to account for dummy letter
    min_base = ord('a') - 2 # start from index 2, 0 for short string and 1 for space in strings
    for i in range(len(count)):
        count[i] = []
    for item in lst:  # generate Counts
        # get column letter if within string, else use dummy position of 0
        if col < len(item):
            letter = ord(item[col]) - min_base
            if letter < 0: # space in the hobby
                letter = 1
        else: # for the length is shorter than column
            letter = 0
        count[letter].append(item)
    index = 0
    for i in range(len(count)):
        frequency = count[i]
        for j in range(len(frequency)):
            output[index] = count[i][j]
            index = index + 1
    return output


def radix_sort_letters(lst, max_col=None):
    '''
    This function is a radix for sorting the strings list
    Reference: https://stackoverflow.com/questions/60968950/radix-sort-for-strings-in-python
    :param lst: the list of hobbies set need to be sorted
    :param max_col: the maximum column of the string in input list
    :return: a sorted list
    complexity is O(27 * M) = O(M) where M is the maximum number of characters
    among all sets of liked things.
    '''
    if not max_col:
        max_col = len(max(lst, key=len))  # edit to max length

    for col in range(max_col - 1, -1, -1):  # max_len-1, max_len-2, ...0
        lst = count_sort_letters(lst, len(lst), col, 26)
    return lst

def interest_groups(data):
    '''
    This function group the people with same interests
    :param data: a list with tuple presenting names and his/her interests
    :return: the list of names who have same interests
    Complexity is O(NM) where N is the number of elements in data,
    M is the maximum number of characters among all sets of liked things
    '''
    temp = {}
    for i in data:
        hobbies = tuple(radix_sort_letters(i[1]))
        if hobbies not in temp:
            # if the sorted hobbies not in temp means this is a new hobbies set
            # if in the list, just append the name to the hobbies set
            temp[hobbies] = []
        temp[hobbies].append(i[0]) # append the name
    names = list(temp.values()) # get the names have the same hobbies in dictionary
    index = 0
    for name in names:
        names[index] = radix_sort_letters(name) # make the name in order
        index += 1
    return names

# data = [("nuka", ["birds", "napping"]),
# ("hadley", ["napping birds", "nash equilibria"]),
# ("yaffe", ["rainy evenings", "the colour red", "birds"]),
# ("laurie", ["napping", "birds"]),
# ("kamalani", ["birds", "rainy evenings", "the colour red"])]
# print(interest_groups(data))
