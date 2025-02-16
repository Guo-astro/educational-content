from PIL import Image, ImageDraw, ImageFont


def create_card_image(output_filename="card_image.png"):
    # 1) Create a blank image
    width, height = 600, 400
    background_color = (63, 81, 181)  # Example: a blueish color
    image = Image.new("RGB", (width, height), background_color)

    # 2) Initialize drawing context
    draw = ImageDraw.Draw(image)

    # 3) (Optional) Load a custom TrueType font
    #    If you have a .ttf file, you can do:
    #       font = ImageFont.truetype("path/to/font.ttf", size=24)
    #    Otherwise, use the default PIL bitmap font:
    font_title = ImageFont.load_default()
    font_text = ImageFont.load_default()

    # 4) Draw a "chip" shape (just a gold rectangle for demonstration)
    chip_x1, chip_y1 = 50, 50
    chip_x2, chip_y2 = 150, 100
    chip_color = (212, 175, 55)  # gold-like color
    draw.rectangle([chip_x1, chip_y1, chip_x2, chip_y2], fill=chip_color)

    # 5) Add bank name / card title
    draw.text((200, 60), "101BANK", fill="white", font=font_title)

    # 6) Add cardholder name
    draw.text((200, 110), "JOHN SMITH", fill="white", font=font_text)

    # 7) Add card number
    draw.text((50, 300), "4137 8947 1175 5904", fill="white", font=font_text)

    # 8) Add expiration date
    draw.text((380, 300), "VALID THRU 10/19", fill="white", font=font_text)

    # 9) Save image
    image.save(output_filename)
    print(f"Saved card image as: {output_filename}")


if __name__ == "__main__":
    create_card_image("card_image.png")
