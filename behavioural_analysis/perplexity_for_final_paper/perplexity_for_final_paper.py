

from PIL import Image, ImageDraw, ImageFont

path = "perplexity_for_final_paper/"
image_paths = [  # each set of 2 is a col.
  
    path+"COVID_perplexity_plot_olmo.png",
    path+"planes_perplexity_plot_olmo.png",
    path+"olmo_training_corpus_year_frequencies.png",
    path+"COVID_perplexity_plot_llama.png", 
    path+"planes_perplexity_plot_llama.png"
]


# Load images
images = [Image.open(img) for img in image_paths]

# Get dimensions
img_width, img_height = images[0].size  # Assuming all images are the same size
grid_width = img_width * 3
grid_height = img_height * 2



# Resize the smaller image if needed
for i, img in enumerate(images):
    if img.size != (img_width, img_height):  # If it's smaller, resize it
        images[i] = img.resize((img_width, img_height), Image.LANCZOS)



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
grid.save(path+"perplexity.png")



#  FOR ii BASIC IMAGE

# from PIL import Image, ImageDraw, ImageFont

# path = "pyvene_ii_paper/graphs/"
# image_paths = [  # each set of 2 is a col.
#     path+"olmo_1980_b.png", # olmo
#     path+"olmo_1980_m.png", # olmo
#     path+"olmo_1980_a.png", # olmo

#     path+"llama_1980_b.png", # llama
#     path+"llama_1980_m.png", # llama
#     path+"llama_1980_a.png", # llama
# ]


# # Load images
# images = [Image.open(img) for img in image_paths]

# # Get dimensions
# img_width, img_height = images[0].size  # Assuming all images are the same size
# grid_width = img_width * 3
# grid_height = img_height * 2

# # Create a blank image for the grid
# grid = Image.new("RGB", (grid_width, grid_height), "white")

# # Define labels
# labels = [f"({chr(97 + i)})" for i in range(len(images))]  # ['(a)', '(b)', '(c)', ..., '(l)']

# # Define font size and load a default font
# try:
#     font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 140)  # Use a system TTF font with size 130
# except IOError:
#     font = ImageFont.load_default()

# # Paste images and overlay labels
# for idx, (img, label) in enumerate(zip(images, labels)):
#     x_offset = (idx % 3) * img_width
#     y_offset = (idx // 3) * img_height
#     grid.paste(img, (x_offset, y_offset))

#     # Draw label in top-left corner
#     draw = ImageDraw.Draw(grid)
#     text_position = (x_offset + 10, y_offset + 10)
#     draw.text(text_position, label, fill="black", font=font)

# # Save or display the result
# grid.save(path+"ii_basic.png")
