# TASK 1C:
# LM input: "Tell me about Wi-Fi. The year is", "Tell me about washboards. The year is",
# etc. We will ask the LM questions about items/events from different time periods, take
# the hidden state used to generate the first token (which should be a year), and compare
# the hidden state’s similarity with embeddings of all year tokens to plot a similarity
# distribution over all years we consider (i.e. 1800 - 2024). This will help us decipher
# which years the LM uses to describe a time-tied object.

OBJECTS = ["Wi-Fi", "washboards", "The Knickerbocker Trust Company", "COVID", "the Taj Mahal",
             "steam engines", "websites", "telephones", "satellites", "social media", "planes",
             "needles", "alcohol", "video games", "iPhones", "South Sudan", "USSR", "Russia", "India",
             "influenza", "organ transplants", "electricity", "Eiffel Tower", "transistors",
             "Istanbul", "Constantinople", "trains", "pagers", "floppy disks", "telegrams", "VHS tapes",
             "CD racks", "Zepplins", "carriages", "check books", "credit cards", "milk delivery", "ice boxes",
             "swing dancing", "Vaudeville Shows", "imperialism", "the Olympics"
]

# TODO: some of these prompts might not elicit the model's internal time
PHRASING = ["Tell me about ", "Explain to me, what is ", "What is ", "Today I learned about ", 
           "I'd like to know more about ", "Give me an overview of ", "Share some insights on ", 
           "Can you tell me about ", "I'd love to learn more about ", "Help me understand ", 
           "I'm curious about ", "Give me an overview of "] # NOTE: IF YOU CHANGE AMOUNT OF THIS, BE CAREFUL, THINK TWICE

# TODO get hidden state of last token (logits) and compare to all years' embeddings
# gemma might have split all years into digits (decode the year) but other models would likely have the entire year as tokens
# for gemma -- attach "2000" to prompt to force decode and see the probability of each char given the prev (conditional probability)
# start with 1950-2050 range and change range as needed after seeing distribution

def generate_task1c(objs, phrasing):
    with open('task1c/task1c.data', 'w', newline='',) as file:
        for obj in objs:
            for phrase in phrasing:
                prompt = phrase + obj + ". The year is "
                file.write(prompt + "\n")
    
if __name__ == "__main__":
    generate_task1c(objs=OBJECTS, phrasing=PHRASING)
