# Chạy đánh giá video-level (bản sửa ICCIES 2027)

> Script **chưa chạy thử** (viết trên máy không có Python/GPU). Chạy `--selftest` trước.

## 1. Chuẩn bị
2 file `eval_video_level.py` và `video_eval_manifest.yaml` nằm trong `efficientNetb4_es/` (source gốc của paper). Checkpoint của source này chỉ load được bằng code của chính nó (FA-ViT-CNN = backbone `favit_freq_lite`).
Không cần cài thêm thư viện (chỉ dùng numpy, pandas, scikit-learn, PyYAML, torch có sẵn).

## 2. Chạy
```bash
cd efficientNetb4_es
python eval_video_level.py --selftest            # ~vài giây, phải in "SELFTEST OK"
# sửa các chỗ <ĐIỀN> trong video_eval_manifest.yaml
python eval_video_level.py --manifest video_eval_manifest.yaml
```
- Bước predict cần GPU: 5 model × (FF++ val + FF++ test + Celeb-DF + UADFV), khoảng 110k frame mỗi model.
- Dự đoán từng frame được cache trong `results_video_level/frames/`. Nếu chạy lại, script không predict lại.
  Muốn chỉ tính lại thống kê thì thêm `--analyze-only`.
- Bootstrap 2000 lần chạy trên CPU, mất vài phút mỗi dataset.

## 3. Kiểm tra trước khi gửi
Mở `results_video_level/sanity.txt`:
- Số video phải bằng **840 / 518 / 98**. Nếu lệch: có video bị mất hết frame do không detect được mặt, hoặc parse video id sai.
- **Không được có** dòng `[WARN] ... frame AUC ... vs paper`. Nếu có, checkpoint, config, TTA hoặc dữ liệu
  không khớp với lúc lấy số cho paper. Phải sửa trước khi gửi; nếu không sửa được thì ghi chú lại.

## 4. Gửi lại
Nén **cả thư mục `results_video_level/`** và gửi. Trong đó có:

| File | Nội dung |
|---|---|
| `paper_tables.md` | Bảng tóm tắt sẵn để điền vào paper |
| `summary_metrics.csv` | Mọi metric: frame-level, video-level (mean, median), ngưỡng 0.5 và ngưỡng chọn trên FF++ val, kèm CI |
| `comparisons.csv` | So sánh paired: ΔAUC với CI bootstrap, p-value; DeLong cho video-level |
| `per_source_ffpp.csv` | AUC theo từng phương pháp giả mạo trên FF++ test |
| `videos/*.csv`, `frames/*.csv` | Điểm từng video và từng frame, để tính lại khi cần |
| `sanity.txt`, `run_info.json` | Kiểm tra số lượng; phiên bản môi trường và hash checkpoint |

## Quy ước (để ghi vào paper)
- Điểm của một video là **trung bình xác suất fake của các frame**. Nhãn của video lấy theo ngưỡng 0.5, giống frame-level.
  Kết quả với ngưỡng chọn trên **FF++ val** (Youden J) được báo cáo phụ.
  Không có ngưỡng nào được chỉnh trên Celeb-DF hoặc UADFV.
- CI 95% cho video-level AUC: bootstrap 2000 lần theo video, phân tầng theo nhãn.
  CI cho frame-level AUC: bootstrap theo cụm, tức là lấy mẫu lại cả video rồi gộp các frame của video đó.
- So sánh hai model: dùng DeLong paired ở video-level. Ở frame-level không dùng DeLong, vì các frame trong
  cùng video không độc lập; thay vào đó dùng paired cluster bootstrap.

## Chỉ chạy FF++ (đủ 5 phương pháp, có FaceShifter)

Dùng để điền các ô FSh/All còn trống trong paper.

```bash
# root giữ như cũ; FaceShifter nằm trong {val,test}/fake/FaceShifter/<video>/<frame>.jpg
python eval_video_level.py --manifest video_eval_manifest_ffpp.yaml
```

- Nếu thiếu thư mục FaceShifter, preflight dừng ngay, trước khi load model.
- Kết quả nằm trong `results_video_level_ffpp/`. Nén cả thư mục rồi gửi lại.
- `per_source_ffpp.csv` và `paper_tables.md` có AUC theo từng phương pháp kèm 95% CI, và DeLong của từng cặp so sánh trên mỗi phương pháp.
- Có thể lọc dataset từ manifest bất kỳ: `--datasets ffpp_val ffpp_test`.
