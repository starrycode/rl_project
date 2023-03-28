import random

import numpy as np
import matplotlib.pyplot as plt

count = 0
fivetwelve = ["" for i in range(512)]
q_matrix = [[0]*9 for i in range(512)]
lastMove = 0
learning_rate = 0.1
discount = 0.9


def resetBoard(board):
    board = ["_" for i in range(9)]
    return board
# if the board has 3 Xs in a row


def printTheArray(arr, n):
    s = ""

    for i in range(0, n):
        s = s + str(arr[i])

    # print(s)

    global count
    global fivetwelve
    fivetwelve[count] = s
    count = count + 1
# Function to generate all binary strings


def generateAllBinaryStrings(n, arr, i):
    if i == n:
        printTheArray(arr, n)
        return

    # First assign "0" at ith position
    # and try for all other permutations
    # for remaining positions
    arr[i] = 0
    generateAllBinaryStrings(n, arr, i + 1)

    # And then assign "1" at ith position and try for all other permutations
    # for remaining positions
    arr[i] = 1
    generateAllBinaryStrings(n, arr, i + 1)


def initArrays():
    global count
    count = 0
    n = 9
    arr = [None] * n
    # Print all binary strings
    generateAllBinaryStrings(n, arr, 0)
    # print(count)


def boardBinary(board):
    #print("representing board as binary string:")
    s = ""
    for i in range(9):
        if board[i] == 'X':
            s = s + '1'
        else:
            s = s + '0'
    # print(s)
    return s


def getNextState(cur_state, action):
    # Modify cur_state while using the index = action - 1 in that index,
    #  change the char to 1 and return that as a new_state
    replacement_char = '1'
    nextState = cur_state[:action] + replacement_char + cur_state[action+1:]
    return nextState


def binStringToDec(b_string):
    return int(b_string, 2)


def decToBinString(rnum):
    binString = "{0:b}".format(int(rnum)).zfill(9)
    return binString  # string type


def numOfWins(state):
    winCount = 0

    # check row
    if state[0] == state[1] == state[2] == 'X':
        winCount += 1
    if state[3] == state[4] == state[5] == 'X':
        winCount += 1
    if state[6] == state[7] == state[8] == 'X':
        winCount += 1

    # check col
    if state[0] == state[3] == state[6] == 'X':
        winCount += 1
    if state[1] == state[4] == state[7] == 'X':
        winCount += 1
    if state[2] == state[5] == state[8] == 'X':
        winCount += 1

    # check diagnoals
    if state[0] == state[4] == state[8] == 'X':
        winCount += 1
    if state[2] == state[4] == state[6] == 'X':
        winCount += 1
    return winCount


def playGame(board):
    global lastMove

    moves = 0
    while numOfWins(board) == 0:
        action = random.choice(range(9))
        while board[action] == 'X':
            action = random.choice(range(9))
        if board[action] != 'X':
            lastMove = action
            board[action] = 'X'

    s = boardBinary(board)
    cur_state = fivetwelve.index(s)
    b_state = decToBinString(cur_state)
    next_b_state = getNextState(b_state, lastMove)
    next_state = eval('0b' + next_b_state)
    q_matrix[cur_state][lastMove] = q_matrix[cur_state][lastMove] + learning_rate * \
        (calcReward(numOfWins(board)) + discount *
         max(q_matrix[next_state]) - q_matrix[cur_state][action])


def calcReward(wins):
    if wins == 3:
        return 5
    if wins == 2:
        return 4
    if wins == 1:
        return 3
    return 0


def main():
    initArrays()
    board = ["_" for i in range(9)]

    episodes = 10000
    for _ in range(episodes):
        playGame(board)
        board = resetBoard(board)

    plt.imshow(q_matrix, cmap='hot', aspect='auto', interpolation='nearest')
    plt.title(
        "State number vs Action taken on grid, learning rate=0.1, discount=0.9")
    plt.ylabel('State number')
    plt.xlabel('Action taken on grid')
    plt.show()


if __name__ == '__main__':
    main()
