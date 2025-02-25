# TASK 1A:
# LM input: "In 2029 there". We expect the LM to output "was", "is", or "will" as its
# first token. We will sweep years. The first token will help us determine what an LM’s
# internal time.

TEMPLATES = ["In [[YEAR]] there ", "In [[YEAR]], they ", 
             "During the year [[YEAR]], scientists ", "By [[YEAR]], weather ", 
             "Today, in [[YEAR]], literature ", "Yesterday, in [[YEAR]], space travel ", 
             "As of [[YEAR]], it ", "In [[YEAR]], after the school bell, he "]

START_YEAR = 1950 # note: if you change start or end year, you have to change it in the plot code as well 
END_YEAR = 2100

def generate_task1a(start_year:int, end_year:int, templates):
    with open('task1a/task1a.data', 'w', newline='',) as file:
        for template in templates:
            for year in range(start_year, end_year + 1):
                prompt = template[:template.find("[[")] + str(year) + template[template.find("]]")+2:]
                file.write(prompt + ',,,' + str(year) + "\n")
    
if __name__ == "__main__":
    generate_task1a(start_year=START_YEAR, end_year=END_YEAR, templates=TEMPLATES)

# TODO
# logit:  batch size  x  seq len * vocab size
# find where "is"/"was"/"were" are using tokenizer
#   tokenizer.encode(WORD)   
#   " was" != "was"  (different tokens)

# present vs past tense using lexer (package: https://www.nltk.org/) or could manually check
 # can also as gpt to do this classifcation

# can also look into probabilities of tenses in addition to just probabilities of is/was/were (a certain word in all tense)
 # add all past tense words' probabilities to understand the prob of the next token being past tense
 # https://github.com/monolithpl/verb.forms.dictionary/blob/master/csv/verbs-all.csv


# TODO - before proj milestone due
# latent concept in model: what is the current time?
# can start editing the model
#  ROME library or use Jing's library
#    (https://github.com/stanfordnlp/pyvene, https://github.com/stanfordnlp/pyvene/blob/main/tutorials/basic_tutorials/Basic_Intervention.ipynb)
#    do a basic test with swapping base and counterfactual information to see whether our edits work [base vs counterfactual prompt]
#    can change the internal model representation ==> find the way that the current year is being encoded inside of the model
#    try to modify the trace instead of the input token -- this means model editing
#   NOTE we are changing model hidden state (from its counterfactual equivalent) at inference time (must have a base and counterfactual 
#               to test); not changing the model weights themselves