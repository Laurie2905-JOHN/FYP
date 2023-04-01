current_names = ['Example 1.txt Ux', 'Example 1.txt Uy']
NewLeg_name = 'tst,gf'
NewLeg_name_list = NewLeg_name.split(',')
print(NewLeg_name_list[0])

newname_result = {}
for i, current_name in enumerate(current_names):
    parts = current_name.split()
    parts[-1] = NewLeg_name_list[i]
    new_name = " ".join(parts)
    newname_result[current_name] = new_name

print(newname_result)