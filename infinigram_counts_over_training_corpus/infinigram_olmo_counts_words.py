# import requests
# import time
# import matplotlib


# # Define the range of numbers
# words = ["beautiful", "gloomy", "rainy"]

# # Initialize a list to store the results
# counts = []

# # Loop through the numbers
# for w in words:
#     payload = {
#         'index': 'v4_rpj_llama_s4',
#         'query_type': 'count',
#         'query': str(w),
#     }
#     response = requests.post('https://api.infini-gram.io/', json=payload)
#     result = response.json()
#     while 'count' not in result: # ensures we get an entry!
#         response = requests.post('https://api.infini-gram.io/', json=payload)
#         result = response.json()
#         # print("RESULT: " + str(result))
    
#     counts.append(result['count'])
#     time.sleep(0.000005) 

# # Now you can plot the counts
# import matplotlib.pyplot as plt

# plt.plot(words, counts)
# plt.xlabel('Year')
# plt.ylabel('Count in Dolma')
# plt.title('year occurences in olmo\'s training data')
# plt.show()



import requests
import time
import matplotlib.pyplot as plt

# Define the words to query
words = ["beautiful", "gloomy", "rainy"]

# Initialize a list to store the counts
counts = []

# Loop through the words and fetch their counts
for w in words:
    payload = {
        'index': 'v4_rpj_llama_s4',
        'query_type': 'count',
        'query': w,
    }
    response = requests.post('https://api.infini-gram.io/', json=payload)
    result = response.json()
    
    while 'count' not in result:  # Ensure we get a valid count
        print("waiting")
        response = requests.post('https://api.infini-gram.io/', json=payload)
        result = response.json()
    
    counts.append(result['count'])
    time.sleep(0.000005)

# Plot the results
plt.bar(words, counts, color=['blue', 'red', 'green'])
plt.xlabel('Words')
plt.ylabel('Count in Dolma (10^8)')
plt.title("Word Occurrences in Dolma's Training Data")
plt.show()
