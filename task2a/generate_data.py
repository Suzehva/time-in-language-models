# TASK 2A:
# Dataset format: "[year] Who is the president of the United States?", "[year] when is
# the next olympics?". Notice that these are similar to Task 1.1.b, but we sweep years.
# Rather than learning internal time, here we test the model’s temporal associations.


# input format: list of   (prompt, [solns])
TEMPLATES = [("The year is 1950. The current president of the United States is ", ["Harry Truman"]),
             
]


# TODO UPDATE THE FORMAT
#              "[[YEAR]]. The current prime minister of PM_COUNTRY is ", 
#              "[[YEAR]]. The next olympics will take place at the city of ", 
#              "[[YEAR]]. The next olympics is in the year ", 
#              "[[YEAR]]. The most popular author is ", 
#              "[[YEAR]]. The most recent winner of SPORT is ", 
# ]

YEARS = [1950, 1970, 1990, 2010, 2030]  

PRES_COUNTRIES = ["the United States", "Brazil", "South Korea", "Iran", "Nigeria", ]
PM_COUNTRIES = ["Netherlands", "the United Kingdom", "Canada", "India", "Thailand", 
                "Malaysia", "Israel", "Italy", "Australia", "Greece", "Bangladesh"]
SPORTS = [  "the FIFA World Cup", "the ICC Cricket World Cup",
            "the Super Bowl", "the Wimbledon Championships", "the Tour de France",
            "the NBA Finals", "the Formula 1 Grand Prix",
            "the US Open (Tennis)", "the MLB World Series", "the Le Mans 24 Hours"]

def generate_task2a(years, templates):
    with open('task2a/task2a.data', 'w', newline='',) as file:
        for template in templates:
            for year in years: # every 3 years is prolly ok?
                prompt = template[:template.find("[[")] + str(year) + template[template.find("]]")+2:]
                if "PRES_COUNTRY" in prompt:
                    for c in PRES_COUNTRIES:
                        w_prompt = prompt[:prompt.find("PRES_COUNTRY")] + c + template[template.find("PRES_COUNTRY")+len("PRES_COUNTRY"):]
                        file.write(w_prompt + "\n")
                elif "PM_COUNTRY" in prompt:
                    for c in PM_COUNTRIES:
                        w_prompt = prompt[:prompt.find("PM_COUNTRY")] + c + template[template.find("PM_COUNTRY")+len("PM_COUNTRY"):]
                        file.write(w_prompt + "\n")
                elif "SPORT" in prompt:
                    for c in SPORTS:
                        w_prompt = prompt[:prompt.find("SPORT")] + c + template[template.find("SPORT")+len("SPORT"):]
                        file.write(w_prompt + "\n")
                else:
                    file.write(prompt + "\n")

if __name__ == "__main__":
    generate_task2a(years=YEARS, templates=TEMPLATES)

