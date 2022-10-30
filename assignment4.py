import math

class WordGraph:
    """
    Reference: Dr Lim Wern Han's lecture
    """
    def __init__(self, wordList):
        # array
        self.wordList = wordList
        n = len(wordList)
        self.vertices = [None] * n
        for i in range(n):
            self.vertices[i] = Vertex(wordList[i], i)



    def __str__(self):
        returnStr = ""
        for vertex in self.vertices:
            returnStr = returnStr + "Vertex " + str(vertex) + "\n"
        return returnStr

    def check_connection(self):
        """
        add edges for the word ladder, check whether they differ by one character
        :param wordList: a list of word string
        complexity: O(W^2) where w is the number of word in word list
        """
        isNeighbour = False
        i = 0
        for i in range(len(self.wordList)):
            for m in range(i+1, len(self.wordList)):
                isNeighbour = sum(self.wordList[i][n] != self.wordList[m][n] for n in range(len(self.wordList[i]))) <= 1
                if isNeighbour == True:
                    self.add_edges([i,m,1])

    def best_start_word(self, target_words):
        """
         find the index of the word for which the longest word ladder to any of
         the words in target_words is as short as possible
         the index of the word should be return will be the central word to all the
         target words
        :param target_words: the index of the word in the word list
        :return: the index of the central word
        complexity:O(W^3) where W is the number of words in the instance of WordGraph
        since using FloydWarshalls in this function
        """
        self.check_connection()
        matrix = self.FloydWarshalls()
        longest = 0
        for n in range(len(target_words)):
            longest = max(matrix[0][target_words[n]],longest)
        longest_ladder = longest
        index = 0
        for i in range(1,len(matrix)):
            longest = 0
            distance = []
            for n in range(len(target_words)):
                distance.append(matrix[i][target_words[n]])
                if longest == math.inf and matrix[i][target_words[n]] != math.inf:
                    longest = matrix[i][target_words[n]]
                elif longest != math.inf and matrix[i][target_words[n]] == math.inf:
                    longest = longest
                else:
                    longest = max(matrix[i][target_words[n]],longest)
            for m in range(len(distance)):
                allInf = True
                d = math.inf
                if d != distance[m]:
                    if len(distance)!= 1 and distance[m]!= 0:
                        # could not be itself since there are more than one target
                        allInf = False
                    elif len(distance) == 1 and distance[m] == 0:
                        allInf = False
            if longest_ladder > longest and allInf != True:
                index = i
                longest_ladder = longest
            if longest_ladder == math.inf:
                if len(target_words) == 1:
                    # disconnected graph but only one target word which means the best start word is itself
                    index = target_words[0]
                else:
                    index = -1
        return index

    def weighted(self):
        for i in range(len(self.wordList)):
            for m in range(i + 1, len(self.wordList)):
                isNeighbour = sum(self.wordList[i][n] != self.wordList[m][n] for n in range(len(self.wordList[i]))) <= 1
                if isNeighbour == True:
                    for n in range(len(self.wordList[i])):
                        if self.wordList[i][n] != self.wordList[m][n]:
                            w = abs(ord(self.wordList[i][n]) - ord(self.wordList[m][n]))
                            self.add_edges([i,m,w])

    def constrained_ladder(self, start, target, constraint_words):
        """
        The function return a list of index of vertice that has shortest distance start from start end at target and pass by one of the vertices in constraint_words
        :param start: indices of vertice
        :param target:indices of vertice
        :param constraint_words:  a list of indices
        :return: a list of indices of vertices (in order)
        complexity:
        O(Dlog(W))) where D is the number of pairs of words in WordGraph which differ by exactly one letter W is the number of words in WordGraph
        """
        self.weighted()
        minDistance = math.inf
        for i in range(len(constraint_words)):
            distance = self.dijkstra(self.vertices[start],self.vertices[constraint_words[i]]) + self.dijkstra(self.vertices[constraint_words[i]],self.vertices[target])
            if distance < minDistance:
                minDistance = distance
        return minDistance


    def add_edges(self, argv_edges, directed=True):
        u = argv_edges[0]
        v = argv_edges[1]
        w = argv_edges[2]
            # add u to v
        current_edge = Edge(u, v, w)
        current_vertex = self.vertices[u]
        current_vertex.add_edge(current_edge)

        current_edge = Edge(v, u, w)
        current_vertex = self.vertices[v]
        current_vertex.add_edge(current_edge)

    def reset(self):
        for vertex in self.vertices:
            vertex.discovered = False
            vertex.visited = False

    def bfs(self, source):
        """
        starting from source
        adding vertices and edges in the graph
        """
        self.reset()
        source = self.vertices[source]
        source.distance = 0
        source.group = "A"
        return_bfs = []
        discovered = []  # queue, FIFO
        discovered.append(source)
        while len(discovered) > 0:
            u = discovered.pop(0)
            u.visited = True
            return_bfs.append(u)
            for edge in u.edges:
                v = edge.v
                if source == v:
                    # found cycle
                    return v.distance  # return length of cycle
                v = self.vertices[v]
                if v.discovered == True:
                    print("There is an cycle.")  # undirected graph
                if v.discovered == False and v.visited == False:
                    if u.group == "A":
                        v.group = "B"
                    else:
                        v.group = "A"
                    discovered.append(v)
                    v.distance = u.distance + 1
                    v.discovered = True  # discovered v, adding it to queue
        return return_bfs  # return every reachable vertices from the source

    def bfs_unwieghted_distance(self, source):
        """
        starting from source
        adding vertices and edges in the graph
        """
        self.reset()
        discovered = []  # queue, FIFO
        discovered.append(source)
        while len(discovered) > 0:
            u = discovered.pop(0)
            u.visited = True
            for edge in u.edges:
                v = edge.v
                v = self.vertices[v]
                if v.discovered == False and v.visited == False:
                    discovered.append(v)
                    v.discovered = True  # discovered v, adding it to queue
                    v.distance = u.distance + 1
                    v.previous = u
        # backtracking

    def dijkstra(self, source, end):
        """
        starting from source
        adding vertices and edges in the graph
        """
        source.distance = 0
        discovered = Heap(len(self.vertices))  # queue, FIFO
        discovered.add((source.distance, source))  # append(key, data)
        while discovered.__len__() > 0:
            u = discovered.pop()
            u[1].visited = True  # distance is finalised
            if u == end:
                return
            # perform edge relaxation on all adjacent vertices
            for edge in u[1].edges:
                v = edge.v
                if self.vertices[v].discovered == False:  # distance is still inf
                    self.vertices[v].discovered = True  # discovered v, adding it to queue
                    self.vertices[v].distance = u[1].distance + edge.w
                    self.vertices[v].previous = u[1]
                    discovered.add((self.vertices[v].distance, self.vertices[v]))  # it is in the heap, but not yet finalised
                elif self.vertices[v].visited == False:
                    # if found a shorter path, update it
                    if self.vertices[v].distance > u[1].distance + edge.w:
                        # update distance
                        self.vertices[v].distance = u[1].distance + edge.w
                        self.vertices[v].previous = u[1]
                        # update heap
                        discovered.update((self.vertices[v].distance, self.vertices[v]))
        return source.distance



    def dfs(self, source):
        """
        starting from source
        adding vertices and edges in the graph
        """
        return_dfs = []
        discovered = []  # stack, LIFO
        discovered.append(source)
        while len(discovered) > 0:
            u = discovered.pop()  # pop last item
            u.visited = True
            return_dfs.append(u)
            for edge in u.edges:
                v = edge.v
                if v.discovered == False:
                    discovered.append(v)
                    v.discovered = True  # discovered v, adding it to queue
        return return_dfs

    def dfs_recur(self, vertex):
        vertex.visited = True
        for next_vertex in vertex.edges:
            if next_vertex.visited == False:
                self.dfs_recur(next_vertex)

    def BellmanFord(self, source):
        for v in self.vertices:
            if v == source:
                v.distance = 0
            else:
                v.distance = math.inf
            v.predecessor = None

        for i in range(1, len(self.vertices)):
            for edge in self.vertices[i]:
                if edge[0].distance + edge[2] < edge[1].distance:
                    edge[1].distance = edge[0].distance + edge[2]
                    edge[1].predecessor = edge[0]

        # check for negative-weight cycle
        for i in range(len(self.vertices)):
            for edge in self.vertices[i]:
                if edge[0].distance + edge[2] < edge[1].distance:
                    print("Graph contains a negative cycle.")
        return self.vertices[i].distance, self.vertices[i].predecessor

    def reachable(self):
        matrix = [None] * len(self.vertices)
        for n in range(len(self.vertices)):
            matrix[n] = [False] * len(self.vertices)
        for a in range(len(self.vertices)):
            for edge in self.vertices[a].edges:
                matrix[a][edge[1]] = True
        for k in range(len(self.vertices)):
            for i in range(len(self.vertices)):
                for j in range(len(self.vertices)):
                    matrix[i][j] = matrix[i][j] or (matrix[i][k] and matrix[k][j])

    def FloydWarshalls(self):
        matrix = [None] * len(self.vertices)
        for n in range(len(self.vertices)):
            matrix[n] = [math.inf] * len(self.vertices)
            matrix[n][n] = 0
        for a in range(len(self.vertices)):
            for edge in self.vertices[a].edges:
                matrix[a][edge.v] = edge.w
        for k in range(len(self.vertices)):
            for i in range(len(self.vertices)):
                for j in range(len(self.vertices)):
                    matrix[i][j] = min(matrix[i][j], matrix[i][k] + matrix[k][j])
        # if diagonal become negative, negative cycle exist
        return matrix


class Vertex:
    def __init__(self, ID, index):
        # list
        self.ID = ID
        self.edges = []
        self.index = index
        # for traversal
        self.discovered = False
        self.visited = False
        # distance
        self.distance = 0  # unweighted
        # backtracking
        self.previous = None
        self.group = None
        self.predecessor = None

    def add_vertex(self, neighbour):
        if neighbour.ID not in self.edges:
            self.edges.append(neighbour.ID)
            neighbour.edges.append(self.ID)
            # self.edges = sorted(self.edges)
            # neighbour.edges = sorted(neighbour.edges)

    def add_edge(self, edge):
        self.edges.append(edge)

    def __str__(self):
        returnStr = str(self.ID)
        for edge in self.edges:
            returnStr = returnStr + "\n with edges " + str(edge)
        return returnStr

    def added_to_queue(self):
        self.discovered = True

    def visited_node(self):
        self.visited = True


class Edge:
    def __init__(self, u, v, w):
        self.u = u
        self.v = v
        self.w = w

    def __str__(self):
        returnStr = str(self.u) + "," + str(self.v) + "," + str(self.w)
        return returnStr


from typing import Generic
""" Basic class implementation of an array of references for FIT units

The code for the init function is a bit cryptic, so I explain it here in
detail. The instance variables holding the physical array is constructed
using the ctypes library to create a py_object (an object that can hold
a reference to any python object). Note that for each value of length we
have that (length * ctypes.py_object) is a type (e.g., if length=5, it
would be a type called py_object_Array_5). Then (length *
ctypes.py_object)() is equivalent to the initialisation in MIPS of the
space to hold the references.

Note that while I do check the precondition in __init__ (noone else
would), I do not check that of getitem or setitem, since that is already
checked by self.array[index].
"""
__author__ = "Julian Garcia for the __init__ code, Maria Garcia de la Banda for the rest"
__docformat__ = 'reStructuredText'

from ctypes import py_object
from typing import TypeVar, Generic

T = TypeVar('T')


class ArrayR(Generic[T]):
    def __init__(self, length: int) -> None:
        """ Creates an array of references to objects of the given length
        :complexity: O(length) for best/worst case to initialise to None
        :pre: length > 0
        """
        if length <= 0:
            raise ValueError("Array length should be larger than 0.")
        self.array = (length * py_object)() # initialises the space
        self.array[:] =  [None for _ in range(length)]

    def __len__(self) -> int:
        """ Returns the length of the array
        :complexity: O(1)
        """
        return len(self.array)

    def __getitem__(self, index: int) -> T:
        """ Returns the object in position index.
        :complexity: O(1)
        :pre: index in between 0 and length - self.array[] checks it
        """
        return self.array[index]

    def __setitem__(self, index: int, value: T) -> None:
        """ Sets the object in position index to value
        :complexity: O(1)
        :pre: index in between 0 and length - self.array[] checks it
        """
        self.array[index] = value


class Heap(Generic[T]):
    MIN_CAPACITY = 1

    def __init__(self, max_size: int) -> None:
        self.length = 0
        self.the_array = ArrayR(max(self.MIN_CAPACITY, max_size) + 1)

    def __len__(self) -> int:
        return self.length

    def is_full(self) -> bool:
        return self.length + 1 == len(self.the_array)

    def rise(self, k: int) -> None:
        """
        Rise element at index k to its correct position
        :pre: 1<= k <= self.length
        """
        while k > 1 and self.the_array[k][0] < self.the_array[k // 2][0]:
            self.swap(k, k // 2)
            k = k // 2

    def add(self, element: T) -> bool:
        """
        Swaps elements while rising
        """
        has_space_left = not self.is_full()

        if has_space_left:
            self.length += 1
            self.the_array[self.length] = element
            self.rise(self.length)

        return has_space_left

    def rise2(self, k: int, element: T) -> int:
        """
        Rise element at index k to its correct position
        :pre: 1<= k <= self.length
        """
        while k > 1 and element[0] < self.the_array[k // 2][0]:
            self.the_array[k] = self.the_array[k // 2]
            k = k//2
        return k

    def add2(self, element: T) -> bool:
        """
        Alternative implementation using shuffling to create
        a hole to perform only one swap at the end
        """
        has_space_left = not self.is_full()
        if has_space_left:
            self.length += 1
            self.the_array[self.rise2(self.length, element)] = element
        return has_space_left

    def add3(self, element: T) -> bool:
        """
        Combined into one method
        More efficient but less readable
        """
        has_space_left = not self.is_full()

        if has_space_left:
            self.length += 1
            k = self.length
            while k > 1 and element[0] > self.the_array[k // 2][0]:
                self.the_array[k] = self.the_array[k // 2]
                k = k // 2

            self.the_array[k] = element

        return has_space_left

    def update(self, element):
        for n in range(self.the_array):
            if self.the_array[n][1] == element[1]:
                self.rise2(n, element)

    def smallest_child(self, k: int) -> int:
        """
        Returns the index of the smallest child of k.
        pre: 2*k <= self.length (at least one child)
        """
        if 2 * k == self.length or self.the_array[2 * k][0] < self.the_array[2 * k + 1][0]:
            return 2*k
        else:
            return 2*k+1

    def pop(self):
        item = self.smallest_child(len(self.the_array))
        self.the_array.array.delete(item)
        return item

    def sink(self, k: int) -> None:
        """ Make the element at index k sink to the correct position """
        while 2*k <= self.length:
            child = self.largest_child(k)
            if self.the_array[k][0] <= self.the_array[child][0]:
                break
            self.swap(child, k)
            k = child

    def create_heap(self, max_size: int, an_array: ArrayR[T] = None) -> None:
        """
        If elements are known in advance, they are in an_array
        Assume that max_size=len(an_array) if given
        """
        self.the_array = ArrayR(max(self.MIN_CAPACITY, max_size) + 1)
        self.length = max_size

        if an_array is not None:
            # copy an_array to self.the_array (shift by 1)
            for i in range(self.length):
                self.the_array[i+1] = an_array[i]

            # heapify every parent
            for i in range(max_size//2, 0, -1):
                self.sink(i)


if __name__ == "__main__":
    words =  ['aaa','bbb','bab','aaf','aaz','baz','caa','cac','dac','dad','ead','eae','bae','abf','bbf']
    myGraph = WordGraph(words)
    start = 0
    end = 1
    detour = [12]
    # print(myGraph.constrained_ladder(start, end, detour))
    print(myGraph)
    print(myGraph.dijkstra(myGraph.vertices[start],myGraph.vertices[detour[0]]))
    print(myGraph.dijkstra(myGraph.vertices[detour[0]], myGraph.vertices[end]))

    # disconnected = ["ggggg", "ggzgg", "abbbb", "ggjgg", "ggzgj", "agzgj", "bbbbb", "bbbbd", "bbbbc"]
    # myGraph = WordGraph(disconnected)
    # print(myGraph)
    # print(myGraph.FloydWarshalls())
    # print(myGraph.best_start_word([2, 7, 8]))

    # NO_EDGES = WordGraph(["aa", "bb", "dd", "zz", "xx", "ff", "uu", "oo"])
    # print(NO_EDGES)
    # print(NO_EDGES.FloydWarshalls())
    # print(NO_EDGES.best_start_word([1, 2, 0, 7, 4, 5]))