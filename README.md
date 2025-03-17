### Winter 2025 | CS224N Final Project
#### Aditi Bhaskar (aditijb@cs.stanford.edu) and Suze van Adrichem (suzeva@cs.stanford.edu)

##### Goal. 

Time is a concept deeply embedded in natural language. In this project, we explore language models' understanding and usage of time and try to localize time in language models. 
We find that language models possess an inherent sense of the current time—approximately 2003 for LLaMA and 2010 for OLMo for complicated settings and 2022 for both in simpler ones. Additionally, we find that models are capable of placing objects and events within specific timeframes and can reason using both their current time and the prompt time even when counterfactual information is presented. However, they exhibit poor performance in tasks measuring temporal associations.

We then try to localize time in language models using Pyvene. We find that layers 0-11 and 0-12 in the block output of the token representing time for Llama and OLMo respectively seem the most plausible options. We also find that replacing time with location has has a similar pattern, suggesting that time and location (and maybe more) share the same representation space. 
