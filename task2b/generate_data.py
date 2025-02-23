# TASK 2B:
# Dataset format: "The president of the United States is Barack Obama. Who was the
# president [year] years before Obama?". This dataset will assess the LM’s temporal
# reasoning (ability to reason about relative time).


PROMPTS = [ "The president of the US during the invention of the telephone was ",
            "The president of the US 10 years before John F Kennedy was ",
            "When Hawaii officially became a state, the largest country in the world was ",
            "The president of the US when the Wright brothers first flew was ",
            "When the Titanic sank, the reigning monarch of the United Kingdom was ",
            "When the Great Fire of London happened, the most powerful empire in the world was ",
            "When the Eiffel Tower was completed, the most populous country in the world was ",
            "The dominant religion in India 1,000 years ago was ",
            "When the Great Wall of China was first built, the dominant empire in the world was ",
            "When Gandhi led the Salt March, the most populous country in the world was ",
            "When paper money was first used in China, the primary form of currency in Europe was ",
            ]

# TODO figure out a way to sweep

def generate_task2b(prompts):
    with open('task2b/task2b.data', 'w', newline='',) as file:
        for prompt in prompts:
            file.write(prompt + "\n")
    
if __name__ == "__main__":
    generate_task2b(prompts=PROMPTS)
