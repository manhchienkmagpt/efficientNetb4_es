# random_copy.py

from pathlib import Path
import random
import shutil
from tqdm import tqdm

# =========================
# Cấu hình
# =========================
SRC_DIR = Path(r"D:\duong_huy_ct7\deepfake-data\train\original")
DST_DIR = Path(r"F:\DeepFakedata\chien_duy_thinh_output\gan_data\input_gan_real\src\original1")
NUM_IMAGES = 24000
SEED = None  # Đặt None nếu không cần tái lập kết quả

# Các định dạng ảnh hỗ trợ
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    # Tạo thư mục đích nếu chưa tồn tại
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # Lấy danh sách ảnh
    images = [
        f for f in SRC_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    total = len(images)
    print(f"Tìm thấy {total:,} ảnh.")

    if total == 0:
        print("Không tìm thấy ảnh nào.")
        return

    # Đặt seed để kết quả có thể tái lập
    if SEED is not None:
        random.seed(SEED)

    # Nếu số lượng yêu cầu lớn hơn số ảnh hiện có
    num_to_copy = min(NUM_IMAGES, total)

    # Chọn ngẫu nhiên
    selected = random.sample(images, num_to_copy)

    print(f"Chuẩn bị copy {num_to_copy:,} ảnh...")

    # Copy ảnh với thanh tiến trình
    for img in tqdm(
        selected,
        desc="Copying images",
        unit="img",
        ncols=100
    ):
        shutil.copy2(img, DST_DIR)

    print(f"\nĐã copy {num_to_copy:,} ảnh vào:")
    print(DST_DIR)


if __name__ == "__main__":
    main()