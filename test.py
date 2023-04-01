list1 = ['1',' \n', '2',' \n',  '3']
list2 = ['4',' \n', '5',' \n',  '6']

# Join the two lists element-wise and concatenate the strings with comma separator
result = [x.strip() + ',' + y.strip() if x.strip() and y.strip() else '' for x, y in zip(list1, list2)]

# Combine the resulting list into a single string
output = '\n'.join(result)

print(output)