# TASK 2A:
# Dataset format: "[year] Who is the president of the United States?", "[year] when is
# the next olympics?". Notice that these are similar to Task 1.1.b, but we sweep years.
# Rather than learning internal time, here we test the model’s temporal associations.

PROMPTS = [
    # US presidents
    ("The year is 1838. The current president of the United States is ", "Martin Van Buren"),
    ("The year is 1870. The current president of the United States is ", "Ulysses S. Grant"),
    ("The year is 1914. The current president of the United States is ", "Woodrow Wilson"),
    ("The year is 1950. The current president of the United States is ", "Harry Truman"),
    ("The year is 1975. The current president of the United States is ", "Gerald Ford"),
    ("The year is 2015. The current president of the United States is ", "Barack Obama"),
    # Summer olympics 
    ("The year is 1950. The next Summer Olympics will take place in the city of ", "Helsinki, Finland"),
    ("The year is 1960. The current Summer Olympics will take place in the city of ", "Rome, Italy"),
    ("The year is 1970. The next Summer Olympics will take place in the city of ", "Munich, West Germany"),
    ("The year is 1979. The previous Summer Olympics had taken place in the city of ", "Montreal, Canada"),
    ("The year is 1998. The next Summer Olympics will take place in the city of ", "Sydney, Australia, in 2000"),
    ("The year is 2006. The next Summer Olympics will take place in the city of ", "Beijing, China, in 2008"),
    ("The year is 2017. The previous Summer Olympics will take place in the city of ", "Rio de Janeiro, Brazil, in 2016"),
    ("The year is 2028. The current Summer Olympics will take place in the city of ", "LA, USA"),
    # international PMs 
    ("The year is 1950. The current prime minister of the Netherlands is ", "Willem Drees"),
    ("The year is 1975. The current prime minister of the Netherlands is ", "Joop den Uyl"),
    ("The year is 2015. The current prime minister of the Netherlands is ", "Mark Rutte"),
    ("The year is 1975. The current prime minister of India is ", "Indira Gandhi"),
    ("The year is 2006. The current prime minister of India is ", "Manmohan Singh"),
    # formula 1 
    ("The year is 1953. This year's winner of the Formula 1 Grand Prix is ", "Alberto Ascari"),
    ("The year is 1968. This year's winner of the Formula 1 Grand Prix is ", "Graham Hill"),
    ("The year is 1983. This year's winner of the Formula 1 Grand Prix is ", "Nelson Piquet"),
    ("The year is 1999. This year's winner of the Formula 1 Grand Prix is ", "Mika Häkkinen"),
    ("The year is 2003. This year's winner of the Formula 1 Grand Prix is ", "Michael Schumacher"),
    ("The year is 2017. This year's winner of the Formula 1 Grand Prix is ", "Lewis Hamilton"),
]


def generate_task2a(prompts):
    with open('task2a/task2a-with-solns.data', 'w', newline='',) as file:
        for text, value in prompts:
            file.write(text + ",,, " + value + "\n")  ## ",,," seperated entries
    with open('task2a/task2a.data', 'w', newline='',) as file:
        for text, value in prompts:
            file.write(text + "\n")  ## just the prompts

if __name__ == "__main__":
    generate_task2a(prompts=PROMPTS)

