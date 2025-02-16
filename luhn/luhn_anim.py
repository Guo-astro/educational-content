import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation

# -------------------------------------------------------
# 1. LOAD IMAGE AND SET UP FIGURE
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

# Read a local card image file; adjust path as needed
# e.g., place "card_image.png" in the same directory
card_img = mpimg.imread("card_image.png")

# Display the image
# 'extent' controls how the image is scaled on the axes
ax.imshow(card_img, extent=[0, 10, 0, 6])
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")  # Hide axes for a clean look

# -------------------------------------------------------
# 2. SET UP CARD NUMBER DIGITS
# -------------------------------------------------------
# Example card number
card_number_str = "4137 8947 1175 5904"

# Extract digits (ignore spaces)
digits = [d for d in card_number_str if d.isdigit()]

# Manually specify (x, y) positions for each digit
# Adjust these values so the digits overlay nicely on your card image
digit_positions = [
    (3.0, 2.4), (3.2, 2.4), (3.4, 2.4), (3.6, 2.4),
    (4.0, 2.4), (4.2, 2.4), (4.4, 2.4), (4.6, 2.4),
    (5.0, 2.4), (5.2, 2.4), (5.4, 2.4), (5.6, 2.4),
    (6.0, 2.4), (6.2, 2.4), (6.4, 2.4), (6.6, 2.4)
]

# Create a text object for each digit
digit_texts = []
for i, digit in enumerate(digits):
    txt = ax.text(
        digit_positions[i][0],
        digit_positions[i][1],
        digit,
        fontsize=14,
        color='white',
        ha='center',
        va='center',
        fontweight='bold'
    )
    digit_texts.append(txt)

# -------------------------------------------------------
# 3. PREPARE THE LUHN ALGORITHM STEPS
#    (We'll animate each step)
# -------------------------------------------------------
# Convert string digits to int
card_digits = [int(d) for d in digits]

# Reverse the digits for typical Luhn processing
reversed_digits = card_digits[::-1]

# Step-by-step data
# We'll store intermediate lists that show how digits transform
steps_info = []

# --- Step 0: Original digits
steps_info.append({
    "description": "Original Digits",
    "highlight_indices": [],
    "processed_digits": card_digits[:]  # no change
})

# --- Step 1: Double every second digit from right
temp1 = []
highlight_indices_step1 = []
for i, val in enumerate(reversed_digits):
    if i % 2 == 1:
        highlight_indices_step1.append(len(reversed_digits) - 1 - i)
        temp1.append(val * 2)
    else:
        temp1.append(val)

temp1 = temp1[::-1]  # reverse back to normal order
steps_info.append({
    "description": "Step 1: Double every second digit (right to left).",
    "highlight_indices": highlight_indices_step1,
    "processed_digits": temp1
})

# --- Step 2: If result > 9, subtract 9 (or sum digits)
temp2 = []
highlight_indices_step2 = []
for i, val in enumerate(temp1):
    if val > 9:
        highlight_indices_step2.append(i)
        temp2.append(val - 9)
    else:
        temp2.append(val)

steps_info.append({
    "description": "Step 2: If > 9, subtract 9.",
    "highlight_indices": highlight_indices_step2,
    "processed_digits": temp2
})

# --- Step 3: Sum all digits
total_sum = sum(temp2)
steps_info.append({
    "description": f"Step 3: Sum all digits = {total_sum}",
    "highlight_indices": list(range(len(temp2))),
    "processed_digits": temp2
})

# --- Step 4: Check validity
valid = (total_sum % 10 == 0)
steps_info.append({
    "description": f"Step 4: Check validity → {'VALID' if valid else 'INVALID'}",
    "highlight_indices": [],
    "processed_digits": temp2
})

# -------------------------------------------------------
# 4. ANIMATION FUNCTION
# -------------------------------------------------------
description_text = ax.text(0.5, 0.5, "",
                           fontsize=12, color='yellow',
                           ha='center', va='center',
                           transform=ax.transAxes)


def update(frame):
    """
    Update function for each animation frame.
    Each frame corresponds to a step in 'steps_info'.
    """
    step_data = steps_info[frame]

    # Update the description text
    description_text.set_text(step_data["description"])

    # Which digits to highlight this step
    highlight_indices = step_data["highlight_indices"]

    # Current digits (post-processing)
    current_digits = step_data["processed_digits"]

    # Update each digit text
    for i, txt in enumerate(digit_texts):
        new_str = str(current_digits[i])
        txt.set_text(new_str)

        # Retrieve the base position for this digit
        base_x, base_y = digit_positions[i]

        # If the number has more than one digit, move it upward and change alignment
        if len(new_str) > 1:
            offset_y = 0.3  # Adjust this offset to your liking
            txt.set_position((base_x, base_y + offset_y))
            txt.set_verticalalignment('bottom')
        else:
            # For single digits, revert to the original position and center alignment
            txt.set_position((base_x, base_y))
            txt.set_verticalalignment('center')

        # Apply highlight color and weight if needed
        if i in highlight_indices:
            txt.set_color('yellow')
            txt.set_fontweight('bold')
        else:
            txt.set_color('white')
            txt.set_fontweight('normal')

    return digit_texts + [description_text]


# -------------------------------------------------------
# 5. CREATE ANIMATION
# -------------------------------------------------------
ani = FuncAnimation(
    fig,
    update,
    frames=len(steps_info),
    interval=5000,  # 2 seconds per step
    blit=False,
    repeat=False
)

plt.tight_layout()
plt.show()
