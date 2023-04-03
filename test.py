# if n_clicks is None:
#     raise PreventUpdate
#
# if "newfile" == ctx.triggered_id:
#
#     new_prb = newData[0]
#
#     new_filenames = newData[1]
#
#     prb = data[0]
#
#     filenames = data[1]
#
#     list1 = [1, 2, 3, 4]
#     list2 = [3, 4, 5, 6]
#
#     # Create a new list to hold the combined values
#     Combined_filenames = filenames.copy()
#     Combined_prb = filenames.copy()
#     indices = []
#
#     # Loop through each value in list2
#     for i, value in enumerate(new_filenames):
#         # Check if the value is already in the combined list
#         if value not in Combined_filenames:
#             # If it's not, add it to the end of the list
#             Combined_filenames.append(value)
#             indices.append(len(Combined_filenames) - 1 + i)
#
#     for i in indices:
#         combined_list.insert(i, values[i])


list1 = ['a', 'b']
list2 = ['k', 'b', 'e', 'd']

# Create a new list to hold the combined values
combined_list = list1.copy()

# Create a list to hold the indices of the unique values
indices = []

# Create a list of additional values to include based on the indices


values = {'a': {}, 'b':{}}
combined_values = values.copy()
values1 = {'k': {}, 'e': {}, 'd': {}}


# Loop through each value in list2
for i, value in enumerate(list2):
    print(i)
    # Check if the value is already in the combined list
    if value not in combined_list:
        # If it's not, add it to the end of the list and record its index
        values[value]=values1[value]
        combined_list.append(value)


print(values)


print(combined_list)



