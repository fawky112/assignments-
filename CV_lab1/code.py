
import cv2
import numpy as np
import os

os.makedirs("results", exist_ok=True)



# Task 1: Selective Colour Isolation


img = cv2.imread("apple.png")

if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    b, g, r = cv2.split(img)

    mask1 = cv2.threshold(r, 100, 255, cv2.THRESH_BINARY)[1]

    mask2 = cv2.compare(
        r.astype(np.int16),
        g.astype(np.int16) + 15,
        cv2.CMP_GT
    )

    mask3 = cv2.compare(
        r.astype(np.int16),
        b.astype(np.int16) + 15,
        cv2.CMP_GT
    )

    mask = cv2.bitwise_and(mask1, mask2)
    mask = cv2.bitwise_and(mask, mask3)

    red_apple = cv2.bitwise_and(img, img, mask=mask)

    gray_background = cv2.bitwise_and(
        gray_bgr,
        gray_bgr,
        mask=cv2.bitwise_not(mask)
    )

    result1 = cv2.add(red_apple, gray_background)

    cv2.imshow("Task 1 - Original Apple", img)
    cv2.imshow("Task 1 - Grayscale", gray_bgr)
    cv2.imshow("Task 1 - Red Mask", mask)
    cv2.imshow("Task 1 - Selective Colour", result1)

    cv2.imwrite("results/task1_grayscale.png", gray_bgr)
    cv2.imwrite("results/task1_mask.png", mask)
    cv2.imwrite("results/task1_result.png", result1)

else:
    print("Error: Could not load apple.png")

# Task 2: Creating a Ghostly Overlay

img1 = cv2.imread("cameraman.jpg")
img2 = cv2.imread("zebra.png")

if img1 is not None and img2 is not None:
    img2 = cv2.resize(
        img2,
        (img1.shape[1], img1.shape[0])
    )

    alpha = 0.5
    beta = 0.5
    gamma = 0

    result2 = cv2.addWeighted(
        img1,
        alpha,
        img2,
        beta,
        gamma
    )

    cv2.imshow("Task 2 - Cameraman", img1)
    cv2.imshow("Task 2 - Zebra", img2)
    cv2.imshow("Task 2 - Ghostly Overlay", result2)

    cv2.imwrite("results/task2_ghost.png", result2)

else:
    print("Error: Could not load cameraman.jpg or zebra.png")


# Task 3: Artistic Vignette Filtering

img = cv2.imread("marsrover.png")

if img is not None:
    height, width = img.shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)

    center = (width // 2, height // 2)
    radius = min(height, width) // 2

    cv2.circle(
        mask,
        center,
        radius,
        255,
        -1
    )

    mask = cv2.GaussianBlur(
        mask,
        (101, 101),
        50
    )

    mask_float = mask.astype(np.float32) / 255.0

    result3 = (
        img.astype(np.float32) *
        mask_float[:, :, np.newaxis]
    ).clip(0, 255).astype(np.uint8)

    cv2.imshow("Task 3 - Original Mars Rover", img)
    cv2.imshow("Task 3 - Vignette Mask", mask)
    cv2.imshow("Task 3 - Vignette Effect", result3)

    cv2.imwrite("results/task3_mask.png", mask)
    cv2.imwrite("results/task3_vignette.png", result3)

else:
    print("Error: Could not load marsrover.png")


# Task 4: Channel Swapping (Surreal Colours)

img = cv2.imread("zebra.png")

if img is not None:
    b, g, r = cv2.split(img)

    result4 = cv2.merge([r, b, g])

    cv2.imshow("Task 4 - Original Zebra", img)
    cv2.imshow("Task 4 - Channel Swapping", result4)

    cv2.imwrite("results/task4_channel_swap.png", result4)

else:
    print("Error: Could not load zebra.png")


# Display and Exit

print("\nAll tasks completed!")
print("Output images saved in the results folder.")

cv2.waitKey(0)
cv2.destroyAllWindows()
