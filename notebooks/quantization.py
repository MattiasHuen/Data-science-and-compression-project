import numpy as np
from PIL import Image


class JPEGCodec2:
    def __init__(self, block_size: int = 8, color_space: str = "YCbCr", q_y: int = 10.0, q_c: int = 16.0):
        self.N = block_size
        self.color_space = color_space
        self.C = self._dct_matrix(block_size)

        self.q_y = q_y
        self.q_c = q_c

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
    
    def quantize_blocks(self, blocks, qsteps):
        return np.round(blocks / qsteps).astype(np.int32)
    
    def dequantize_blocks(self, blocks, qsteps):
        return (blocks * qsteps).astype(np.float64)
    
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
    
    def quantize(self, transformed: dict) -> dict:
        if self.color_space == "YCbCr":
            q1 = self.q_y
            q2 = q3 = self.q_c
        else:
            q1 = q2 = q3 = self.q_y

        return {
            "color_space": self.color_space,
            "shape": transformed["shape"],
            "ch1": self.quantize_blocks(transformed["ch1"], q1),
            "ch2": self.quantize_blocks(transformed["ch2"], q2),
            "ch3": self.quantize_blocks(transformed["ch3"], q3)
        }
    
    def dequantize(self, quantized: dict) -> dict:
        if self.color_space == "YCbCr":
            q1 = self.q_y
            q2 = q3 = self.q_c
        else:
            q1 = q2 = q3 = self.q_y

        return {
            "color_space": self.color_space,
            "shape": quantized["shape"],
            "ch1": self.dequantize_blocks(quantized["ch1"], q1),
            "ch2": self.dequantize_blocks(quantized["ch2"], q2),
            "ch3": self.dequantize_blocks(quantized["ch3"], q3)
        }

    def encode(self, image: Image.Image) -> dict:
        transformed = self.transform(image)
        quantized = self.quantize(transformed)
        return quantized
    
    def decode(self, encoded: dict) -> Image.Image:
        dequantized = self.dequantize(encoded)
        decoded = self.inverse_transform(dequantized)
        return decoded
