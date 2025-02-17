# TASK 1A:
# LM input: "In 2029 there". We expect the LM to output "was", "is", or "will" as its
# first token. We will sweep years. The first token will help us determine what an LM’s
# internal time.

templates = ["In [[YEAR]] there "]
years = ["1980", "2000", "2020", "2040"]

with open('task1a.data', 'w', newline='') as txtfile:
    for template in templates:
        for year in years:
            txtfile.write(template[:template.find("[[")] + year + template[template.find("]]")+2:] + "\n")
