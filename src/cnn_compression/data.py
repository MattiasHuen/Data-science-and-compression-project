from pathlib import Path

import typer
from torch.utils.data import Dataset
from collections import Counter
from constants import *
from PIL import Image
import matplotlib.pyplot as plt


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = Path(data_path / "raw")
        self.image_paths = sorted(self.data_path.rglob("*.png"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        return {
            "image": image,
            "path": image_path,
        }

    def get_image_stats(self) -> dict:
        widths = []
        heights = []
        stats = Counter()

        for image_path in self.image_paths:
            with Image.open(image_path) as img:
                width, height = img.size
            widths.append(width)
            heights.append(height)
            stats[(width, height)] += 1

        if not widths or not heights:
            return {
                "num_images": 0,
                "avg_width": 0,
                "avg_height": 0,
                "most_common_size": [],
            }

        return {
            "num_images": len(self.image_paths),
            "unique_sizes": len(stats),
            "min_width": min(widths),
            "max_width": max(widths),
            "mean_width": sum(widths) / len(widths),
            "min_height": min(heights),
            "max_height": max(heights),
            "mean_height": sum(heights) / len(heights),
            "most_common_sizes": stats.most_common(101),
        }

    @staticmethod
    def rotate_to_landscape(image: Image.Image) -> Image.Image:
        """Rotate portrait images so width >= height."""
        if image.height > image.width:
            image = image.rotate(90, expand=True)
        return image

    @staticmethod
    def center_crop(image: Image.Image, crop_size: tuple[int, int]) -> Image.Image:
        """Center crop image to (width, height)."""
        crop_w, crop_h = crop_size
        img_w, img_h = image.size

        if img_w < crop_w or img_h < crop_h:
            raise ValueError(
                f"Image too small for requested crop. "
                f"Image size: {(img_w, img_h)}, crop size: {crop_size}"
            )

        left = (img_w - crop_w) // 2
        top = (img_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h

        return image.crop((left, top, right, bottom))

    def preprocess(
        self,
        output_folder: Path,
        crop_size: tuple[int, int] = (2032, 1344),
    ) -> None:
        """
        Preprocess raw data and save to output folder.

        Steps:
        1. Convert to RGB
        2. Rotate portrait images to landscape
        3. Center-crop to crop_size
        4. Save as PNG
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        skipped = []

        for image_path in self.image_paths:
            try:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    img = self.rotate_to_landscape(img)
                    img = self.center_crop(img, crop_size)

                    output_path = output_folder / image_path.name
                    img.save(output_path)

            except Exception as e:
                skipped.append((image_path, str(e)))

        print(f"Saved processed images to: {output_folder}")
        print(f"Processed: {len(self.image_paths) - len(skipped)}")
        print(f"Skipped: {len(skipped)}")

        if skipped:
            print("\nSkipped files:")
            for path, err in skipped:
                print(f"- {path}: {err}")


    
def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder=Path("src/Dataset/processed"), crop_size=(2032, 1344))
    stats = dataset.get_image_stats()
    print(stats)


if __name__ == "__main__":
    output_folder = DATA_FOLDER / "processed"
    tester = preprocess(DATA_FOLDER, output_folder)
