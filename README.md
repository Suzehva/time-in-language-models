### Winter 2025 | CS224N Final Project
#### Aditi Bhaskar (aditijb@cs.stanford.edu) and Suze van Adrichem (suzeva@cs.stanford.edu)

##### Goal. 

Time is a concept that is deeply embedded in natural language. Knowing the current time is
so critical for day-to-day conversations that newer chat models explicitly specify the current date in
their system prompts. In this project, we explore time representation in language models (LM’s) -
specifically, the conflict between an LM’s internal and in-context times.

We consider three questions: (1) Can we determine a LM’s internal "current" time? (2) Where is
time localized in LM’s? (3) Can we edit an LM to control time-related outputs? The first of these
questions will help us develop a conceptual and qualitative understanding of time in relationship to
LM’s and will lay the foundation for the second two. We will consider time from the perspective of
temporal association, how an LM understands factual information at absolute times, and temporal
reasoning, how an LM reasons about timelines and times relative to each other.

Finding answers to the second and third question is exciting because it could advance our understand-
ing of how time is represented internally in LM’s, which could be helpful in determining strategies to
make LM’s better at time-related tasks. We will utilize causal tracing and interchange interventions 
(as outlined in Stanford's pyvene https://github.com/stanfordnlp/pyvene) to explore this.
