# TASK 1A:
# LM input: "In 2029 there". We expect the LM to output "was", "is", or "will" as its
# first token. We will sweep years. The first token will help us determine what an LM’s
# internal time.

TEMPLATES = ["In [[YEAR]] there ", "In [[YEAR]], they ", 
             "During the year [[YEAR]], scientists ", "By [[YEAR]], weather ", 
             "Today, in [[YEAR]], literature ", "Yesterday, in [[YEAR]], space travel ", 
             "As of [[YEAR]], it ", "In [[YEAR]], after the school bell, he "]

START_YEAR = 1950
END_YEAR = 2050

def generate_year_data(start_year:int, end_year:int, templates):
    with open('task1a.data', 'w', newline='',) as txtfile:
        for template in templates:
            for year in range(start_year, end_year + 1):
                txtfile.write(template[:template.find("[[")] + str(year) + template[template.find("]]")+2:] + "\n")

if __name__ == "__main__":
    generate_year_data(start_year=START_YEAR, end_year=END_YEAR, templates=TEMPLATES)
