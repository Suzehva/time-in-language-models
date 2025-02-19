# TASK 1D:
# LM input: "The year is 2001. Tell me about COVID-19". We will ask the LM
# counterfactual questions (i.e. questions where it’s internal time might conflict with the
# time given in the prompt) to analyze how the LM answers. In the example, if it explains
# COVID-19, the model uses its internal time, but if it explains that COVID-19 doesn’t
# exist at this point (or a similar response), it means the model uses the in-context time.

OBJECTS = ["Wi-Fi", "washboards", "The Knickerbocker Trust Company", "COVID", "the Taj Mahal",
             "steam engines", "websites", "telephones", "satellites", "social media", "planes",
             "needles", "alcohol", "video games", "iPhones", "South Sudan", "USSR", "Russia", "India",
             "influenza", "organ transplants", "electricity", "Eiffel Tower", "transistors",
             "Istanbul", "Constantinople", "trains", "pagers", "floppy disks", "telegrams", "VHS tapes",
             "CD racks", "Zepplins", "carriages", "check books", "credit cards", "milk delivery", "ice boxes",
             "swing dancing", "Vaudeville Shows", "imperialism", "the Olympics"
            ]

# TODO: we could alternatively keep the tense constant
# fix the tense to match obj
TENSE = ["is ", "was ", "will be "]

START_YEAR = 1950
END_YEAR = 2050

def generate_obj_given_year_data(start_year, end_year, objs, tense):
    with open('task1d/task1d.data', 'w', newline='',) as file:
        for yr in range(start_year, end_year, 3): # TODO: is every 3 years ok?
            for obj in objs:
                for t in tense:
                    prompt = "The year is " + str(yr) + ". " + obj + " " + t
                    file.write(prompt + "\n")
    
if __name__ == "__main__":
    generate_obj_given_year_data(start_year=START_YEAR, end_year=END_YEAR, objs=OBJECTS, tense=TENSE)
