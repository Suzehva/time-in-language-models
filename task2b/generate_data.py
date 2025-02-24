# TASK 2B:
# Dataset format: "The president of the United States is Barack Obama. Who was the
# president [year] years before Obama?". This dataset will assess the LM’s temporal
# reasoning (ability to reason about relative time).

# input format: list of   (prompt, [solns])

# PROMPTS = [ ("The president of the US during the invention of the telephone was ", ["TODO"]),
#             ("The president of the US 10 years before John F Kennedy was ",["TODO"]),
#             ("When Hawaii officially became a state, the largest country in the world was ",["TODO"]),
#             ("The president of the US when the Wright brothers first flew was ",["TODO"]),
#             ("When the Titanic sank, the reigning monarch of the United Kingdom was ",["TODO"]),
#             ("When the Great Fire of London happened, the most powerful empire in the world was ",["TODO"]),
#             ("When the Eiffel Tower was completed, the most populous country in the world was ",["TODO"]),
#             ("The dominant religion in India 1,000 years ago was ",["TODO"]),
#             ("When the Great Wall of China was first built, the dominant empire in the world was ",["TODO"]),
#             ("When Gandhi led the Salt March, the most populous country in the world was ",["TODO"]),
#             ("When paper money was first used in China, the primary form of currency in Europe was ",["TODO"])
#             ]

PROMPTS = [
    ("The president of the US during the invention of the telephone was ", ["Ulysses S. Grant"]),
    ("The president of the US 10 years before John F. Kennedy was ", ["Harry S. Truman"]),
    ("When Hawaii officially became a state, the largest country in the world was ", ["Soviet Union"]),
    ("The president of the US when the Wright brothers first flew was ", ["Theodore Roosevelt"]),
    ("When the Titanic sank, the reigning monarch of the United Kingdom was ", ["King George V"]),
    ("When the Great Fire of London happened, the most powerful empire in the world was ", ["Ottoman Empire"]),
    ("When the Eiffel Tower was completed, the most populous country in the world was ", ["China"]),
    ("The dominant religion in India 1,000 years ago was ", ["Hinduism"]),
    ("When the Great Wall of China was first built, the dominant empire in the world was ", ["Achaemenid Empire"]),
    ("When Gandhi led the Salt March, the most populous country in the world was ", ["China"]),
    ("When paper money was first used in China, the primary form of currency in Europe was ", ["Gold and silver coins", "Neo-Assyrian Empire", "Median Empire"])
]

def generate_task2b(prompts):
    with open('task2b/task2b-with-solns.data', 'w', newline='',) as file:
        for text, values in prompts:
            file.write(text)
            for v in values:
                file.write(",,," + v) ## ",,," seperated entries
            file.write("\n")
    with open('task2b/task2b.data', 'w', newline='',) as file:
        for text, value in prompts:
            file.write(text + "\n")  ## just the prompts
 
if __name__ == "__main__":
    generate_task2b(prompts=PROMPTS)
