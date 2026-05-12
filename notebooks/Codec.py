import numpy as np
from PIL import Image
from collections import Counter


class JPEGCodec3:
    def __init__(self, block_size: int = 8, color_space: str = "YCbCr", q_y: int = 10.0, q_c: int = 16.0):
        self.N = block_size
        self.color_space = color_space
        self.C = self._dct_matrix(block_size)

        self.q_y = q_y
        self.q_c = q_c

        self.zz_indices = self._zigzag_indices()

    # ------------------------------------------------------------------
    # DCT matrix
    # ------------------------------------------------------------------
    def _dct_matrix(self, N: int) -> np.ndarray:
        C = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                alpha = np.sqrt(1 / N) if i == 0 else np.sqrt(2 / N)
                C[i, j] = alpha * np.cos(((2 * j + 1) * i * np.pi) / (2 * N))
        return C
    
    # ------------------------------------------------------------------
    # Block splitting / merging
    # ------------------------------------------------------------------    
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
    
    # ------------------------------------------------------------------
    # DCT / inverse DCT
    # ------------------------------------------------------------------    
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
    
    # ------------------------------------------------------------------
    # Quantization / dequantization
    # ------------------------------------------------------------------    
    def quantize_blocks(self, blocks, qsteps):
        return np.round(blocks / qsteps).astype(np.int32)
    
    def dequantize_blocks(self, blocks, qsteps):
        return (blocks * qsteps).astype(np.float64)
    
    # ------------------------------------------------------------------
    # Image preprocessing / postprocessing
    # ------------------------------------------------------------------    
    def preprocess_image(self, image):
        arr = np.asarray(image.convert(self.color_space), dtype=np.uint8).astype(np.float64)
        return arr - 128.0
    
    def postprocess_image(self, arr):  
        arr = np.clip(arr + 128.0, 0, 255).astype(np.uint8)
        if self.color_space == "YCbCr":
            return Image.fromarray(arr, mode="YCbCr").convert("RGB")
        return arr
    
    # ------------------------------------------------------------------
    # Forward transform / inverse transform
    # ------------------------------------------------------------------
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
    
    # ------------------------------------------------------------------
    # Quantize full transformed image
    # ------------------------------------------------------------------    
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
    
    # ------------------------------------------------------------------
    # Original simple encode / decode
    # -----------------------------------------------------------------
    def encode(self, image: Image.Image) -> dict:
        transformed = self.transform(image)
        quantized = self.quantize(transformed)
        return quantized
    
    def decode(self, encoded: dict) -> Image.Image:
        dequantized = self.dequantize(encoded)
        decoded = self.inverse_transform(dequantized)
        return decoded
    
    # ------------------------------------------------------------------
    # Zig-zag scan
    # ------------------------------------------------------------------
    def _zigzag_indices(self):
        """
        Returns the standard zig-zag order for an NxN block.

        This orders coefficients from low spatial frequency to high
        spatial frequency. After quantization, the high-frequency part
        usually contains many zeros, which makes run-length coding useful.
        """
        indices = []
        N = self.N

        for s in range(2 * N - 1):
            if s % 2 == 0:
                for i in range(s, -1, -1):
                    j = s - i
                    if i < N and j < N:
                        indices.append((i, j))
            else:
                for j in range(s, -1, -1):
                    i = s - j
                    if i < N and j < N:
                        indices.append((i, j))

        return indices
    
    def zigzag_block(self, block: np.ndarray) -> np.ndarray:
        return np.array(
            [block[i, j] for i, j in self.zz_indices],
            dtype=np.int32
        )
    
    def inverse_zigzag_block(self, coeffs: np.ndarray) -> np.ndarray:
        block = np.zeros((self.N, self.N), dtype=np.int32)

        for value, (i, j) in zip(coeffs, self.zz_indices):
            block[i, j] = value

        return block
    
    # ------------------------------------------------------------------
    # DC DPCM
    # ------------------------------------------------------------------

    def dpcm_encode_dc(self, dc_values) -> np.ndarray:
        """
        Encode DC values as differences between neighboring blocks.

        JPEG does this because neighboring blocks often have similar
        average intensity.
        """
        dc_values = np.asarray(dc_values, dtype=np.int32)

        diffs = np.empty_like(dc_values)
        previous = 0

        for i, dc in enumerate(dc_values):
            diffs[i] = dc - previous
            previous = dc

        return diffs

    def dpcm_decode_dc(self, dc_diffs) -> np.ndarray:
        dc_diffs = np.asarray(dc_diffs, dtype=np.int32)

        dc_values = np.empty_like(dc_diffs)
        previous = 0

        for i, diff in enumerate(dc_diffs):
            dc_values[i] = previous + diff
            previous = dc_values[i]

        return dc_values

    # ------------------------------------------------------------------
    # AC run-length coding
    # ------------------------------------------------------------------

    def rle_encode_ac(self, ac: np.ndarray) -> list:
        """
        Run-length encode AC coefficients.

        Each nonzero coefficient is represented as:

            (number_of_preceding_zeros, value)

        The block ends with EOB.
        """
        symbols = []
        zero_run = 0

        for value in ac:
            value = int(value)

            if value == 0:
                zero_run += 1
            else:
                symbols.append((zero_run, value))
                zero_run = 0

        symbols.append(("EOB",))

        return symbols

    def rle_decode_ac(self, symbols: list) -> np.ndarray:
        ac = []

        for symbol in symbols:
            if symbol == ("EOB",):
                break

            zero_run, value = symbol
            ac.extend([0] * zero_run)
            ac.append(value)

        # AC part must have exactly 63 coefficients for an 8x8 block.
        target_length = self.N * self.N - 1
        ac.extend([0] * (target_length - len(ac)))

        return np.array(ac[:target_length], dtype=np.int32)

    # ------------------------------------------------------------------
    # Entropy-symbol preparation for one channel
    # ------------------------------------------------------------------

    def entropy_prepare_channel(self, q_blocks: np.ndarray) -> dict:
        """
        Convert quantized DCT blocks into JPEG-like symbols.

        For each block:
        - zig-zag scan
        - store DC coefficient
        - run-length encode AC coefficients

        After all blocks:
        - DPCM encode the DC sequence
        """
        nb_h, nb_w, _, _ = q_blocks.shape

        dc_values = []
        ac_symbols = []

        for i in range(nb_h):
            for j in range(nb_w):
                zz = self.zigzag_block(q_blocks[i, j])

                dc_values.append(int(zz[0]))
                ac_symbols.append(self.rle_encode_ac(zz[1:]))

        dc_diffs = self.dpcm_encode_dc(dc_values)

        return {
            "dc": dc_diffs,
            "ac": ac_symbols,
            "block_shape": (nb_h, nb_w),
        }

    def entropy_reconstruct_channel(self, encoded_channel: dict) -> np.ndarray:
        """
        Reconstruct quantized DCT blocks from DC DPCM and AC RLE symbols.
        """
        nb_h, nb_w = encoded_channel["block_shape"]

        dc_values = self.dpcm_decode_dc(encoded_channel["dc"])
        ac_symbols = encoded_channel["ac"]

        blocks = np.empty((nb_h, nb_w, self.N, self.N), dtype=np.int32)

        k = 0
        for i in range(nb_h):
            for j in range(nb_w):
                dc = dc_values[k]
                ac = self.rle_decode_ac(ac_symbols[k])

                zz = np.concatenate([[dc], ac])
                blocks[i, j] = self.inverse_zigzag_block(zz)

                k += 1

        return blocks

    # ------------------------------------------------------------------
    # JPEG-like symbol encode / decode
    # ------------------------------------------------------------------

    def encode_symbols(self, image: Image.Image) -> dict:
        """
        Full JPEG-like encoder up to entropy-symbol generation.

        This does not yet output a real binary JPEG file.
        It outputs the symbols that would normally be Huffman or
        arithmetic coded.
        """
        transformed = self.transform(image)
        quantized = self.quantize(transformed)

        return {
            "color_space": self.color_space,
            "shape": quantized["shape"],
            "ch1": self.entropy_prepare_channel(quantized["ch1"]),
            "ch2": self.entropy_prepare_channel(quantized["ch2"]),
            "ch3": self.entropy_prepare_channel(quantized["ch3"]),
        }

    def decode_symbols(self, encoded: dict) -> Image.Image:
        """
        Decode from JPEG-like symbols back to an image.
        """
        q_ch1 = self.entropy_reconstruct_channel(encoded["ch1"])
        q_ch2 = self.entropy_reconstruct_channel(encoded["ch2"])
        q_ch3 = self.entropy_reconstruct_channel(encoded["ch3"])

        quantized = {
            "color_space": encoded["color_space"],
            "shape": encoded["shape"],
            "ch1": q_ch1,
            "ch2": q_ch2,
            "ch3": q_ch3,
        }

        return self.decode(quantized)

    # ------------------------------------------------------------------
    # Entropy / estimated rate
    # ------------------------------------------------------------------

    def collect_entropy_symbols(self, encoded: dict) -> list:
        """
        Collect all DC and AC symbols into one list for entropy estimation.
        """
        symbols = []

        for ch in ["ch1", "ch2", "ch3"]:
            channel = encoded[ch]

            for dc_diff in channel["dc"]:
                symbols.append(("DC", int(dc_diff)))

            for block_symbols in channel["ac"]:
                for symbol in block_symbols:
                    if symbol == ("EOB",):
                        symbols.append(("AC", "EOB"))
                    else:
                        run, value = symbol
                        symbols.append(("AC", int(run), int(value)))

        return symbols

    def entropy_from_symbols(self, symbols: list) -> float:
        """
        Empirical entropy in bits/symbol.
        """
        counts = Counter(symbols)
        total = sum(counts.values())

        probs = np.array([count / total for count in counts.values()])
        entropy = -np.sum(probs * np.log2(probs))

        return float(entropy)

    def estimate_rate(self, encoded: dict) -> dict:
        """
        Estimate entropy, total bits, and bits per pixel.

        This is not an actual Huffman bitstream length.
        It is an entropy-based estimate.
        """
        symbols = self.collect_entropy_symbols(encoded)

        H = self.entropy_from_symbols(symbols)
        estimated_bits = H * len(symbols)

        H_img, W_img, _ = encoded["shape"]
        bpp = estimated_bits / (H_img * W_img)

        return {
            "entropy_bits_per_symbol": H,
            "num_symbols": len(symbols),
            "estimated_bits": estimated_bits,
            "estimated_bpp": bpp,
        }

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------

    def mse(self, original: Image.Image, reconstructed: Image.Image) -> float:
        x = np.asarray(original.convert("RGB"), dtype=np.float64)
        y = np.asarray(reconstructed.convert("RGB"), dtype=np.float64)

        return float(np.mean((x - y) ** 2))

    def psnr(self, original: Image.Image, reconstructed: Image.Image) -> float:
        mse_value = self.mse(original, reconstructed)

        if mse_value == 0:
            return float("inf")

        return float(10 * np.log10((255.0 ** 2) / mse_value))
