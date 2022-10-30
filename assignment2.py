# Author: Shen Qiu Tong
# Student ID: 30763819
# Task 1

def count_encounters(target_difficulty, monster_list):
    """
    This function find all possible sets that makes the sum of monster
    difficulties equal to target_difficulty.

    :param target_difficulty: the difficulty of monster we want to target
    :param monster_list: a list with monster type and their difficulties
    :return: the number of possibility set of monsters' difficultes sum up to target_difficulty
    complexity: O(DM) where D is the value of target_difficulty
    M is the length of monster_list
    Reference: http://leetcode.libaoj.in/combination-sum.html
    """
    difficulties = []
    for i in range(len(monster_list)):
        difficulties.append(monster_list[i][1])
    # get the difficulty of the monster
    # difficulties = sorted(difficulties) # sort it
    res = []
    def combinationToSum(remainder, combination, start):
        if remainder == 0:
            # a group of combination is finished
            res.append(list(combination))
            return
        elif remainder < 0:
            # more than the target_difficulty, stop exploration.
            return
        for i in range(start, len(difficulties)):
            # add the number into the combination
            combination.append(difficulties[i])
            # give the current number another chance, rather than moving on
            combinationToSum(remainder - difficulties[i], combination, i)
            # backtracking, remove the number from the combination
            # a group of combination is finished
            # the sum not equal to the target_difficulty
            combination.pop()
    combinationToSum(target_difficulty, [], 0)
    # start process with original  target, empty list, and start from the first difficulty
    count = len(res) # amount of all possibilities
    return count

# Task 2
def best_lamp_allocation(num_p, num_l, probs):
    """
    This function calculates the highest probability of the plants given by the num_l of lamps
    :param num_p: number of plants
    :param num_l: number of lamps provided
    :param probs: the possibility of plant growing list given the provided lamps
    :return: the highest possibility of  being ready by the party that can be
    obtained by allocating lamps to plants optimally
    complexity: O(PL^2) where p is the num_p and L is num_l
    Reference: https://stackoverflow.com/questions/69013586/how-elements-from-lists-are-chosen
    """
    memo = [None] * num_p
    for i in range(len(memo)):
        memo[i] = [None] *(num_l + 1) # plus one since a plant can have 0 lamp
    p = 0
    l = 0 # l represent the number of lamp given to the specific plant (memo[i])
    for j in range(num_l + 1):
        # find the highest probability of the plant[0] among all the given amount of lamps, and find the number of lamps it used
        if probs[0][j] > p:
            p = probs[0][j]
            l = j
        memo[0][j] = (p, l)
    for i in range(1, len(memo)):
        # calculate the probability for all plants that given 0 lamp
        p = memo[i - 1][0][0] * probs[i][0]
        l = 0
        memo[i][0] = (p, l)
    for i in range(1, len(memo)):
        for j in range(1, len(memo[0])):
            # given j lamps to all plants
            prev = memo[i - 1]
            new_p = 0
            new_l = 0
            m = j
            while m > 0:
                prevPlan = prev[m]
                restLamp = j - prevPlan[1]
                n = 0
                while n <= restLamp:
                    # If there are lamps left or just used up
                    if prevPlan[0] * probs[i][n] > new_p:
                        # if the possibility is higher than previous plan, then update the possibility and lamp used
                        new_p = prevPlan[0] * probs[i][n]
                        new_l = n + prevPlan[1]
                    n += 1
                m -= 1
            after_p = memo[i][j - 1][0]
            after_l = memo[i][j - 1][1]
            # compare to the possibility of not using the added lamps, choose the plan that higher probability
            if new_p > after_p:
                actual_p = new_p
                actual_l = new_l
            else:
                actual_p = after_p
                actual_l = after_l
            memo[i][j] = (actual_p, actual_l) # updated
    return memo[-1][-1][0]