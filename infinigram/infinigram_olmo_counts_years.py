import requests
import time
import matplotlib


# Define the range of numbers
numbers = range(1500, 2201)

# Initialize a list to store the results
counts = []

# Loop through the numbers
for num in numbers:
    payload = {
        'index': 'v4_rpj_llama_s4',
        'query_type': 'count',
        'query': str(num),
    }
    response = requests.post('https://api.infini-gram.io/', json=payload)
    result = response.json()
    while 'count' not in result: # ensures we get an entry!
        response = requests.post('https://api.infini-gram.io/', json=payload)
        result = response.json()
        # print("RESULT: " + str(result))
    
    counts.append(result['count'])
    time.sleep(0.000005) 

# Now you can plot the counts
import matplotlib.pyplot as plt

plt.plot(numbers, counts)
plt.xlabel('Year')
plt.ylabel('Count in Dolma')
plt.title('year occurences in olmo\'s training data')
plt.show()

