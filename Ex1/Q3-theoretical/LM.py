import numpy as np
def get_most_probable_path(dp, length, starting_letter):
    max = 0
    most_probable_word = ''
    for letter in range(3):
        if matrix[starting_letter][letter] * dp[length-1][letter][0] > max:
            max = matrix[starting_letter][letter] * dp[length-1][letter][0]
            if starting_letter == 0:
                most_probable_word = 'B' + dp[length-1][letter][1]
            elif starting_letter == 1:
                most_probable_word = 'K' + dp[length-1][letter][1]
            else:
                most_probable_word = 'O' + dp[length-1][letter][1]
    return (max, most_probable_word)


# Create the matrix using numpy array
matrix = np.array([[0.1, 0.325, 0.25, 0.325],
                   [0.4, 0, 0.4, 0.2],
                   [0.2, 0.2, 0.2, 0.4],
                   [1, 0, 0, 0]])


#create a 3 by 5 matrix of zeros without using numpy
dp = [[0 for i in range(3)] for j in range(5)]
dp[0] = [(matrix[0][3],'B'), (matrix[1][3], 'K'), (matrix[2][3], 'O')]
for i in range(1, 5):
    for j in range(3):
        dp[i][j] = get_most_probable_path(dp, i, j)

#print the matrix
for i in range(5):
    print(dp[i])
        

