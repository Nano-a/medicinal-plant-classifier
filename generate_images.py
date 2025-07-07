from PIL import Image, ImageDraw, ImageFont
import os

def create_gradient_image(width, height, color1, color2, filename):
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    for y in range(height):
        r = int(color1[0] + (color2[0] - color1[0]) * y / height)
        g = int(color1[1] + (color2[1] - color1[1]) * y / height)
        b = int(color1[2] + (color2[2] - color1[2]) * y / height)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    img.save(filename)

def create_plant_image(filename):
    img = Image.new('RGB', (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw plant elements
    draw.ellipse((150, 200, 250, 300), fill=(85, 107, 47))  # Leaf
    draw.ellipse((170, 220, 230, 280), fill=(255, 255, 255))  # Flower
    draw.line((200, 300, 200, 400), fill=(139, 69, 19), width=10)  # Stem
    
    img.save(filename)

# Create images
current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(current_dir, 'static', 'images')
os.makedirs(images_dir, exist_ok=True)

create_plant_image(os.path.join(images_dir, 'plant-hero.png'))
create_gradient_image(1920, 1080, (52, 152, 219), (44, 62, 80), os.path.join(images_dir, 'hero-bg.png'))
create_gradient_image(1920, 1080, (46, 204, 113), (39, 174, 96), os.path.join(images_dir, 'result-bg.png'))
