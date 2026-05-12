import numpy as np
from PIL import Image


class JPEGCodec1:
    def __init__(self, block_size: int = 8, color_space: str = "YCbCr"):
        self.N = block_size
        self.color_space = color_space
        self.C = self._dct_matrix(block_size)

    def _dct_matrix(self, N: int) -> np.ndarray:
        C = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                alpha = np.sqrt(1 / N) if i == 0 else np.sqrt(2 / N)
                C[i, j] = alpha * np.cos(((2 * j + 1) * i * np.pi) / (2 * N))
        return C
    
    def _split_blocks(self, channel):
        H, W = channel.shape
        assert H % self.N == 0 and W % self.N == 0, "Image dimensions must be divisible by block size"
        return channel.reshape(
            H // self.N, self.N,
            W // self.N, self.N
        ).transpose(0, 2, 1, 3)

    def _merge_blocks(self, blocks):
        nb_h, nb_w, bh, bw = blocks.shape
        return blocks.transpose(0, 2, 1, 3).reshape(nb_h * bh, nb_w * bw)
    
    def _dct2(self, block):
        return self.C @ block @ self.C.T
    
    def _idct2(self, block):
        return self.C.T @ block @ self.C
    
    def transform_blocks(self, blocks):
        out = np.empty_like(blocks, dtype=np.float64)
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                out[i, j] = self._dct2(blocks[i, j])
        return out
    
    def inverse_transform_blocks(self, blocks):
        out = np.empty_like(blocks, dtype=np.float64)
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                out[i, j] = self._idct2(blocks[i, j])
        return out
    
    def preprocess_image(self, image):
        arr = np.asarray(image.convert(self.color_space), dtype=np.uint8).astype(np.float64)
        return arr - 128.0
    
    def postprocess_image(self, arr):  
        arr = np.clip(arr + 128.0, 0, 255).astype(np.uint8)
        if self.color_space == "YCbCr":
            return Image.fromarray(arr, mode="YCbCr").convert("RGB")
        return arr
    
    def transform(self, image: Image.Image) -> dict:
        arr = self.preprocess_image(image)
        channels = []

        for c in range(3):
            blocks = self._split_blocks(arr[:, :, c])
            dct_blocks = self.transform_blocks(blocks)
            channels.append(dct_blocks)

        return {
            "color_space": self.color_space,
            "shape": arr.shape,
            "ch1": channels[0],
            "ch2": channels[1],
            "ch3": channels[2]
        }

    def inverse_transform(self, encoded: dict) -> Image.Image:
        rec_channels = []

        for key in ["ch1", "ch2", "ch3"]:
            blocks = self.inverse_transform_blocks(encoded[key])
            rec_channels.append(self._merge_blocks(blocks))
        
        rec = np.stack(rec_channels, axis=-1)
        return self.postprocess_image(rec)
    
    def encode(self, image: Image.Image) -> dict:
        return self.transform(image)
    
    def decode(self, encoded: dict) -> Image.Image:
        return self.inverse_transform(encoded)


















# def DCT_block(N: int = 8) -> np.ndarray:
#     """
#     Construct the orthonormal DCT matrix of size NxN.
#     """
#     C = np.zeros((N, N), dtype=np.float64)

#     for i in range(N):
#         for j in range(N):
#             alpha = np.sqrt(1 / N) if i == 0 else np.sqrt(2 / N)
#             C[i, j] = alpha * np.cos(((2 * j + 1) * i * np.pi) / (2 * N))

#     return C


# def merge_blocks(blocks: np.ndarray) -> np.ndarray:
#     nb_h, nb_w, bh, bw = blocks.shape
#     return blocks.transpose(0, 2, 1, 3).reshape(nb_h * bh, nb_w * bw)


# def dct2(block: np.ndarray, C: np.ndarray) -> np.ndarray:
#     """
#     Apply 2D DCT to one NxN block using matrix multiplication.
#     """
#     return C @ block @ C.T

# def idct2(block: np.ndarray, C: np.ndarray) -> np.ndarray:
#     """
#     Apply 2D inverse DCT to one NxN block using matrix multiplication.
#     """
#     return C.T @ block @ C

# def TransformBlocks(blocks: np.ndarray, C: np.ndarray) -> np.ndarray:
#     """
#     Apply DCT to every block.
#     """
#     nb_h, nb_w, _, _ = blocks.shape
#     out = np.empty_like(blocks, dtype=np.float64)

#     for i in range(nb_h):
#         for j in range(nb_w):
#             out[i, j] = dct2(blocks[i, j], C)

#     return out

# def ITransformBlocks(blocks: np.ndarray, C: np.ndarray) -> np.ndarray:
#     """
#     Apply inverse DCT to every block.
#     """
#     nb_h, nb_w, _, _ = blocks.shape
#     out = np.empty_like(blocks, dtype=np.float64)

#     for i in range(nb_h):
#         for j in range(nb_w):
#             out[i, j] = idct2(blocks[i, j], C)

#     return out


# def channel_blocks(channel: np.ndarray, block_size: int = 8) -> np.ndarray:
#     H, W = channel.shape

#     assert H % block_size == 0, f"Height {H} is not divisible by {block_size}"
#     assert W % block_size == 0, f"Width {W} is not divisible by {block_size}"

#     blocks = channel.reshape(
#         H // block_size, block_size,
#         W // block_size, block_size
#     ).transpose(0, 2, 1, 3)

#     return blocks


# def Transform(image: Image.Image, color: str = "RGB", N: int = 8) -> dict:
#     """
#     Apply blockwise DCT to a 3-channel image.
#     Returns DCT coefficients for each channel.
#     """
#     C = DCT_block(N)

#     if color == "RGB":
#         image_np = np.asarray(image.convert("RGB"), dtype=np.uint8)
#         ch1 = image_np[:, :, 0].astype(np.float64) - 128.0
#         ch2 = image_np[:, :, 1].astype(np.float64) - 128.0
#         ch3 = image_np[:, :, 2].astype(np.float64) - 128.0
#         channel_names = ("R", "G", "B")

#     elif color == "YCbCr":
#         image_np = np.asarray(image.convert("YCbCr"), dtype=np.uint8)
#         ch1 = image_np[:, :, 0].astype(np.float64) - 128.0
#         ch2 = image_np[:, :, 1].astype(np.float64) - 128.0
#         ch3 = image_np[:, :, 2].astype(np.float64) - 128.0
#         channel_names = ("Y", "Cb", "Cr")

#     else:
#         raise ValueError("color must be either 'RGB' or 'YCbCr'")

#     ch1_blocks = channel_blocks(ch1, block_size=N)
#     ch2_blocks = channel_blocks(ch2, block_size=N)
#     ch3_blocks = channel_blocks(ch3, block_size=N)

#     ch1_dct = TransformBlocks(ch1_blocks, C)
#     ch2_dct = TransformBlocks(ch2_blocks, C)
#     ch3_dct = TransformBlocks(ch3_blocks, C)

#     return {
#         "color_space": color,
#         "channel_names": channel_names,
#         "C": C,
#         "ch1_blocks": ch1_dct,
#         "ch2_blocks": ch2_dct,
#         "ch3_blocks": ch3_dct,
#         "ch1_full": merge_blocks(ch1_dct),
#         "ch2_full": merge_blocks(ch2_dct),
#         "ch3_full": merge_blocks(ch3_dct),
#         "full_image": np.stack((merge_blocks(ch1_dct), merge_blocks(ch2_dct), merge_blocks(ch3_dct)), axis=-1)
#             }



# def ITransform(image: dict, color: str = "RGB", N: int = 8) -> dict:
#     """
#     Apply blockwise inverse DCT to a 3-channel image.
#     Returns inverse DCT coefficients for each channel.
#     """
#     C = DCT_block(N)
    
#     ch1_blocks = image["ch1_blocks"]
#     ch2_blocks = image["ch2_blocks"]
#     ch3_blocks = image["ch3_blocks"]

#     ch1_idct = ITransformBlocks(ch1_blocks, C)
#     ch2_idct = ITransformBlocks(ch2_blocks, C)
#     ch3_idct = ITransformBlocks(ch3_blocks, C)

#     ch1_full = merge_blocks(ch1_idct) + 128.0
#     ch2_full = merge_blocks(ch2_idct) + 128.0
#     ch3_full = merge_blocks(ch3_idct) + 128.0

#     reconstructed = np.stack((ch1_full, ch2_full, ch3_full), axis=-1).clip(0, 255).astype(np.uint8)

#     if color == "YCbCr":
#         reconstructed_rgb = np.asarray(Image.fromarray(reconstructed, mode="YCbCr").convert("RGB"))
#     else:
#         reconstructed_rgb = reconstructed
    
#     return {
#         "color_space": color,
#         "C": C,
#         "ch1_blocks": ch1_idct,
#         "ch2_blocks": ch2_idct,
#         "ch3_blocks": ch3_idct,
#         "ch1_full": ch1_full,
#         "ch2_full": ch2_full,
#         "ch3_full": ch3_full,
#         "full_image": reconstructed
#             }