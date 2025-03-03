# TASK 2B:
# Dataset format: "The president of the United States is Barack Obama. Who was the
# president [year] years before Obama?". This dataset will assess the LM’s temporal
# reasoning (ability to reason about relative time).

PROMPT_STYLES= [
    ("The current president of the US is [[A]]. The name of the president 10 years ago was"),
    ("[[A]] is the current U.S. president. The name of the U.S. president 10 years ago was"),
    ("The name of the president 10 years before [[A]] was"),
    ("US president [[A]] was inaugurated today. The name of the president in office 10 years ago was"),
]

NOW_VS_10_YR_AGO = [
    # ("George Washington", "None"),  # might be too counterfactual
    ("John Adams", "George Washington"),
    ("Thomas Jefferson", "John Adams"),
    ("James Madison", "Thomas Jefferson"),
    ("James Monroe", "James Madison"),
    ("John Quincy Adams", "James Monroe"),
    ("Andrew Jackson", "James Monroe"),
    ("Martin Van Buren", "Andrew Jackson"),
    ("William Henry Harrison", "Andrew Jackson"),
    ("James K. Polk", "Andrew Jackson"),
    ("Zachary Taylor", "Martin Van Buren"),
    ("Franklin Pierce", "John Tyler"),
    # ("James Buchanan", "Millard Fillmore"),  # Fillmore finished Taylor’s term, technically. maybe too much of a technicality to ask small models
    # ("Abraham Lincoln", "Millard Fillmore"),
    ("Ulysses S. Grant", "James Buchanan"),
    ("Rutherford B. Hayes", "Andrew Johnson"),
    ("James A. Garfield", "Ulysses S. Grant"),
    ("Chester A. Arthur", "Ulysses S. Grant"),
    ("Benjamin Harrison", "Rutherford B. Hayes"),
    ("William McKinley", "Grover Cleveland"),
    ("Theodore Roosevelt", "Benjamin Harrison"),
    ("William Howard Taft", "William McKinley"),
    ("Woodrow Wilson", "Theodore Roosevelt"),
    ("Warren G. Harding", "William Howard Taft"),
    ("Calvin Coolidge", "Woodrow Wilson"),
    ("Herbert Hoover", "Woodrow Wilson"),
    ("Franklin D. Roosevelt", "Calvin Coolidge"),
    ("Harry S. Truman", "Franklin D. Roosevelt"),
    ("Dwight D. Eisenhower", "Franklin D. Roosevelt"),
    ("John F. Kennedy", "Harry S. Truman"),
    ("Lyndon B. Johnson", "Dwight D. Eisenhower"),
    ("Richard Nixon", "	Dwight D. Eisenhower"),
    ("Gerald Ford", "Lyndon B. Johnson"),
    ("Jimmy Carter", "Richard Nixon"),
    ("Ronald Reagan", "Richard Nixon"),
     
    # the following are likely too recent; also the model defaults to saying "Bill Clinton" so these might give false positives

    # ("George H. W. Bush", "Jimmy Carter"),  # accuracy with George Bush is difficult to compute
    # ("Bill Clinton", "Ronald Reagan"),
    # ("George W. Bush", "George H. W. Bush"),  
    # ("Barack Obama", "George W. Bush"),
    # ("Donald Trump", "Barack Obama"),
    # ("Joe Biden", "Donald Trump"),
]

def replace_placeholder(prompt: str, replacement: str, placeholder: str):
    start = prompt.find(placeholder)
    if start != -1:
        return prompt[:start] + replacement + prompt[start + len(placeholder):]


def generate_task2b(facts, prompts):
    with open('task2b/task2b-with-solns.data', 'w', newline='',) as file_soln:  # prompts with solns
        with open('task2b/task2b.data', 'w', newline='',) as file:  # just the prompts
            for a, b in facts:
                for p in prompts:
                    phrase = replace_placeholder(p, a, "[[A]]")
                    file_soln.write(phrase + " | " + b + "\n")
                    file.write(phrase + "\n")


if __name__ == "__main__":
    generate_task2b(facts = NOW_VS_10_YR_AGO, prompts=PROMPT_STYLES)
