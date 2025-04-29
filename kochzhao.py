import numpy as np
import cv2


def prepare_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    h, w = img.shape
    h = h - (h % 8)
    w = w - (w % 8)
    img = cv2.resize(img, (w, h))

    return img


def embed_message(image, message):
    binary_message = ''.join(format(ord(c), '08b') for c in message)
    msg_len = len(binary_message)

    h, w = image.shape
    blocks = h * w // 64
    if msg_len > blocks:
        raise ValueError("Wiadomość za długa dla tego obrazu!")

    stego_image = image.copy().astype(float)

    bit_idx = 0

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if bit_idx >= msg_len:
                break

            block = stego_image[i:i + 8, j:j + 8]

            dct_block = cv2.dct(block)

            coeff1 = dct_block[4, 1]
            coeff2 = dct_block[3, 2]

            bit = int(binary_message[bit_idx])
            delta = 10.0

            if bit == 1:
                if coeff1 <= coeff2:
                    dct_block[4, 1] = coeff2 + delta
                    dct_block[3, 2] = coeff1 - delta
            else:
                if coeff1 >= coeff2:
                    dct_block[4, 1] = coeff2 - delta
                    dct_block[3, 2] = coeff1 + delta

            modified_block = cv2.idct(dct_block)
            stego_image[i:i + 8, j:j + 8] = modified_block

            bit_idx += 1

    stego_image = np.clip(stego_image, 0, 255).astype(np.uint8)
    return stego_image, msg_len


def extract_message(stego_image, msg_len):
    extracted_bits = []
    h, w = stego_image.shape

    bit_idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if bit_idx >= msg_len:
                break

            block = stego_image[i:i + 8, j:j + 8].astype(float)

            dct_block = cv2.dct(block)

            coeff1 = dct_block[4, 1]
            coeff2 = dct_block[3, 2]

            bit = '1' if coeff1 > coeff2 else '0'
            extracted_bits.append(bit)

            bit_idx += 1

    binary_message = ''.join(extracted_bits)
    message = ''
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i + 8]
        if len(byte) == 8:
            message += chr(int(byte, 2))

    return message



input_image_path = 'input.jpg'
output_image_path = 'stego_output.jpg'
message = "Sekretna wiadomosc"

processed_image = prepare_image(input_image_path)

stego_image, msg_len = embed_message(processed_image, message)

cv2.imwrite(output_image_path, stego_image)

extracted_message = extract_message(stego_image, msg_len)

print(f"Oryginalna wiadomość: {message}")
print(f"Odczytana wiadomość: {extracted_message}")
