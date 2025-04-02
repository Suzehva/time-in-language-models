from PIL import Image, ImageDraw, ImageFont



path = "pyvene_causal_tracing/"
llama = "llama_paper/"
olmo = "olmo_paper/"
image_paths = [  # each set of 4 is a row. it goes olmo/llama/o/l adnd 1980/1980/2030/2030
    path+olmo+"combined_In_1980_on_a_beautiful_day_there_OLMo_block_output2025_03_16_10_12_14__fancy.png",
        path+olmo+"combined_In_1980_on_a_beautiful_day_there_OLMo_mlp_activation2025_03_16_10_12_14__fancy.png",
            path+olmo+"combined_In_1980_on_a_beautiful_day_there_OLMo_attention_output2025_03_16_10_12_14__fancy.png", 


    path+llama+"combined_In_1980_on_a_beautiful_day_there_Llama_block_output2025_03_16_10_12_14__fancy.png",
    # path+olmo+"combined_In_2030_on_a_beautiful_day_there_OLMo_block_output2025_03_16_10_12_14__fancy.png",
    # path+llama+"combined_In_2030_on_a_beautiful_day_there_Llama_block_output2025_03_16_10_12_14__fancy.png",
        
    path+llama+"combined_In_1980_on_a_beautiful_day_there_Llama_mlp_activation2025_03_16_10_12_14__fancy.png",
    # path+olmo+"combined_In_2030_on_a_beautiful_day_there_OLMo_mlp_activation2025_03_16_10_12_14__fancy.png",
    # path+llama+"combined_In_2030_on_a_beautiful_day_there_Llama_mlp_activation2025_03_16_10_12_14__fancy.png",

    path+llama+"combined_In_1980_on_a_beautiful_day_there_Llama_attention_output2025_03_16_10_12_14__fancy.png", 
    # path+olmo+"combined_In_2030_on_a_beautiful_day_there_OLMo_attention_output2025_03_16_10_12_14__fancy.png", 
    # path+llama+"combined_In_1980_on_a_beautiful_day_there_Llama_attention_output2025_03_16_10_12_14__fancy.png", 


        path+olmo+"combined_In_2030_on_a_beautiful_day_there_OLMo_block_output2025_03_16_10_12_14__fancy.png",
        path+olmo+"combined_In_2030_on_a_beautiful_day_there_OLMo_mlp_activation2025_03_16_10_12_14__fancy.png",
            path+olmo+"combined_In_2030_on_a_beautiful_day_there_OLMo_attention_output2025_03_16_10_12_14__fancy.png", 


    path+llama+"combined_In_2030_on_a_beautiful_day_there_Llama_block_output2025_03_16_10_12_14__fancy.png",
        
    # path+olmo+"combined_In_1980_on_a_beautiful_day_there_OLMo_mlp_activation2025_03_16_10_12_14__fancy.png",
    # path+llama+"combined_In_1980_on_a_beautiful_day_there_Llama_mlp_activation2025_03_16_10_12_14__fancy.png",
    path+llama+"combined_In_2030_on_a_beautiful_day_there_Llama_mlp_activation2025_03_16_10_12_14__fancy.png",

    # path+olmo+"combined_In_1980_on_a_beautiful_day_there_OLMo_attention_output2025_03_16_10_12_14__fancy.png", 
    # path+llama+"combined_In_1980_on_a_beautiful_day_there_Llama_attention_output2025_03_16_10_12_14__fancy.png", 
    path+llama+"combined_In_1980_on_a_beautiful_day_there_Llama_attention_output2025_03_16_10_12_14__fancy.png", 

]



# Load images
images = [Image.open(img) for img in image_paths]

# Get dimensions
img_width, img_height = images[0].size  # Assuming all images are the same size
grid_width = img_width * 3
grid_height = img_height * 4

# Create a blank image for the grid
grid = Image.new("RGB", (grid_width, grid_height), "white")

# Define labels
labels = [f"({chr(97 + i)})" for i in range(len(images))]  # ['(a)', '(b)', '(c)', ..., '(l)']

# Define font size and load a default font
try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 140)  # Use a system TTF font with size 130
except IOError:
    font = ImageFont.load_default()

# Paste images and overlay labels
for idx, (img, label) in enumerate(zip(images, labels)):
    x_offset = (idx % 3) * img_width
    y_offset = (idx // 3) * img_height
    grid.paste(img, (x_offset, y_offset))

    # Draw label in top-left corner
    draw = ImageDraw.Draw(grid)
    text_position = (x_offset + 10, y_offset + 10)
    draw.text(text_position, label, fill="black", font=font)

# Save or display the result
grid.save(path+"causal_tracing_paper.png")
# grid.show()
