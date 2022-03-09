import numpy as np

w = np.array([8, 3, 5, 2]) #waga przedmiotów
W = 9 #maksymalna waga plecaka
p = np.array([16, 8, 9, 6]) #wartość przedmiotów

# przegląd wyczerpujący

best_result = 0
best_list = '0000' # w wyniku 1 dla przedmiotów zabranych

for i in range(1, pow(2, len(w))): # tworzenie wszystkich możliwych kombinacji
    i = bin(i)[2:]
    while len(i) < len(w):
        i = '0' + i

    counter = 0
    curr_result = 0
    curr_weight = 0

    for j in i: # sprawdzanie wartości dla każdej kombinacji
        if j == '1':
            curr_weight += w[counter]
            
            if curr_weight > W: break;
            curr_result += p[counter]
        counter += 1
    
    if curr_weight > W: continue
    if curr_result > best_result:
        best_result = curr_result
        best_list = i

print("\nMetoda wyczerpująca: ")
print(best_result)
print(best_list)



# za pomocą heurystyki

s = [] #stosunki wartości do masy poszczególnych przedmiotów
full_list = []

for i in range(len(w)):
    s.append(p[i]/w[i])

for i in range(len(w)):
    full_list.append((s[i], w[i], p[i], i))

full_list.sort()
full_list.reverse() # posortowana malejąco według stosunku p/w lista informacji o przedmiotach

result = 0
weight = 0
best_list_num = []
best_list = ''

for element in full_list:   # dodawanie kolejnych najbardziej opłacalnych wyrazów przy spełnieniu założeń o wadze
    if weight + element[1] > W:
        continue
    weight += element[1]
    result += element[2]
    best_list_num.append(element[3])

for i in range(len(s)):
    if i in best_list_num:
        best_list += '0'
    else: best_list += '1'

best_list = best_list[::-1]

print('\nMetoda heurystyczna:')
print(result)
print(best_list)
