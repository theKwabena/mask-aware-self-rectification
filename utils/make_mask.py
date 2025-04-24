import cv2
import os
import argparse


def draw_mask(image_path, save_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))
    mask = img.copy() * 0
    drawing = False
    radius = 8

    def draw(event, x, y, flags, param):
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(mask, (x, y), radius, (255, 255, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow("Draw Mask (Left Click + Drag)")
    cv2.setMouseCallback("Draw Mask (Left Click + Drag)", draw)

    while True:
        preview = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
        cv2.imshow("Draw Mask (Left Click + Drag)", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit without saving
            break
        elif key == ord('s'):  # Press 's' to save mask
            binary = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(binary, 1, 255, cv2.THRESH_BINARY)
            cv2.imwrite(save_path, binary)
            print(f"Saved mask to {save_path}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to target image")
    parser.add_argument("--out", required=True, help="Path to save the binary mask")
    args = parser.parse_args()

    draw_mask(args.image, args.out)
