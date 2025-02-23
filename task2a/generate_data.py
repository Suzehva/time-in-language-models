# TASK 2A:
# Dataset format: "[year] Who is the president of the United States?", "[year] when is
# the next olympics?". Notice that these are similar to Task 1.1.b, but we sweep years.
# Rather than learning internal time, here we test the model’s temporal associations.


TEMPLATES = ["[[YEAR]]. The current president of the PRES_COUNTRY is ",
             "[[YEAR]]. The current prime minister of the PM_COUNTRY is ", 
             "[[YEAR]]. The next olympics will take place at the city of ", 
             "[[YEAR]]. The next olympics is in the year ", 
             "[[YEAR]]. The most popular author is ", 
             "[[YEAR]]. The most recent winner of FIFA World Cup is ", 
]

START_YEAR = 1950
END_YEAR = 2050

PRES_COUNTRIES = ["United States", "India"]
PM_COUNTRIES = ["Netherlands", "United Kingdom", "Canada"]

def generate_task2a(start_year:int, end_year:int, templates):
    with open('task2a/task2a.data', 'w', newline='',) as file:
        for template in templates:
            for year in range(start_year, end_year + 1):
                prompt = template[:template.find("[[")] + str(year) + template[template.find("]]")+2:]
                if "PRES_COUNTRY" in prompt:
                    for c in PRES_COUNTRIES:
                        w_prompt = prompt[:prompt.find("PRES_COUNTRY")] + c + template[template.find("PRES_COUNTRY")+len("PRES_COUNTRY"):]
                        file.write(w_prompt + "\n")
                elif "PM_COUNTRY" in prompt:
                    for c in PM_COUNTRIES:
                        w_prompt = prompt[:prompt.find("PM_COUNTRY")] + c + template[template.find("PM_COUNTRY")+len("PM_COUNTRY"):]
                        file.write(w_prompt + "\n")
                else:
                    file.write(prompt + "\n")
    
if __name__ == "__main__":
    generate_task2a(start_year=START_YEAR, end_year=END_YEAR, templates=TEMPLATES)

