# Wongnai Review Rating Prediction (Naive Bayes)

โปรเจกต์ทำนายคะแนนดาว (1–5) ของรีวิวร้านอาหารภาษาไทยจาก Wongnai ด้วยโมเดล **Multinomial Naive Bayes**
โดยสร้าง "ตารางคำ" (word × star) เป็นแหล่งที่มาของความน่าจะเป็น P(word | star) ที่โมเดลใช้

## ขั้นตอนการทำงาน

1. **โหลดข้อมูล** `w_review_train.csv` (ไม่มี header, คั่นด้วย `;`, คอลัมน์สุดท้ายคือ rating) เข้า `df` — 40,000 รีวิว
2. **ตัดคำไทย** ด้วย `pythainlp` (engine `newmm`) แล้วตัด stop words ออก
   (เก็บคำที่บอกอารมณ์ไว้: ไม่, มาก, น้อย, สุด, ที่สุด, เกิน, จริง, เลย)
3. **สร้างตารางคำ** — 1 แถวต่อ 1 คำ มีจำนวนคำในแต่ละดาว, P(word | star) แบบ Laplace smoothing 
4. **พิมพ์ Word Count** แยกตามดาวออก Console
5. **Word Cloud** แสดง 1 หน้าต่าง เรียงรูปตามดาว (กรองคำทั่วไปออกเฉพาะตอนวาดภาพ)
6. **แบ่ง train/test 80/20** (แยกตามดาว) แล้วสร้างตารางคำจากชุด train เท่านั้น จากนั้น **คำนวณ Naive Bayes เอง** แสดง Test Accuracy และ Classification Report ที่คำนวณเองเช่นกัน ไม่ใช้ `scikit-learn`
   - feature = คำเดี่ยว + คำคู่ (bigram เช่น `ไม่_อร่อย`) โดยโมเดลใช้คำทุกคำรวม stop words
   - นับแบบมี/ไม่มี (binary) คำเดียวกันในรีวิวเดียวนับครั้งเดียว และตัด feature ที่เจอน้อยกว่า 5 ครั้ง (`MIN_COUNT`), Laplace `ALPHA = 0.5`
   - คะแนนของรีวิว = ค่าเฉลี่ย log P(feature | star) (รีวิวยาวไม่ครอบงำ) + bias ของแต่ละดาว
   - bias ของแต่ละดาวจูนบน validation 20% ที่แบ่งจากชุด train (ตั้ง `OBJECTIVE` เป็น `macro_f1` หรือ `accuracy`)
7. **Heat Map** ของ Confusion Matrix อีก 1 หน้าต่าง
8. **Heat Map ของตารางคำ** P(word | star) สำหรับ 30 คำยอดนิยม อีก 1 หน้าต่าง
9. **เทรนใหม่ด้วยข้อมูลทั้ง 40,000 รีวิว** แล้วทำนาย `review_dataset/test_file.csv` บันทึกเป็น `submission.csv`

## ผลลัพธ์

| เวอร์ชัน | Test Accuracy | Macro F1 | Recall 1 ดาว |
|---|---|---|---|
| เดิม (unigram, log prior) | 0.523 | 0.29 | 0.00 |
| ใหม่ `OBJECTIVE = "macro_f1"` (ค่าเริ่มต้น) | 0.523 | **0.45** | **0.40** |
| ใหม่ `OBJECTIVE = "accuracy"` | **0.557** | 0.38 | 0.16 |

คลาสไม่สมดุล (3 และ 4 ดาวรวมกันเกิน 75%) เวอร์ชันเดิมจึงไม่ทำนาย 1 ดาวเลย
เวอร์ชันใหม่ทำนายได้ทุกดาว โดย accuracy เท่าเดิม แต่ Macro F1 เพิ่มจาก 0.29 เป็น 0.45

## ไฟล์ในโปรเจกต์

| ไฟล์ | คำอธิบาย |
|---|---|
| `wongnai_nb.py` | สคริปต์หลัก ทำทุกขั้นตอนข้างต้น |
| `word_table_by_star.csv` | ตารางคำที่สร้างได้ |
| `wordcloud_by_star.png` | ภาพ Word Cloud แยกตามดาว |
| `word_heatmap.png` | Heat Map ของ P(word \| star) สำหรับ 30 คำยอดนิยม |
| `confusion_matrix.png` | ภาพ Confusion Matrix |
| `submission.csv` | ผลทำนายของ `test_file.csv` (reviewID, rating) |
| `review_dataset/` | ชุดข้อมูล test และ sample submission |

## วิธีรัน

ติดตั้งไลบรารี:

```bash
pip install pythainlp pandas numpy matplotlib seaborn wordcloud
```

ดาวน์โหลด `w_review_train.csv` จาก [wongnai-corpus](https://github.com/wongnai/wongnai-corpus) มาวางใน `review_dataset/`
(หรือตั้ง environment variable `WONGNAI_TRAIN` เป็น path ของไฟล์) แล้วรัน (ไฟล์ผลลัพธ์จะถูกบันทึกในโฟลเดอร์เดียวกับสคริปต์):

```bash
python wongnai_nb.py
```

ใช้เวลาตัดคำประมาณ 1–3 นาที (รวม test_file) แล้วจะมี 3 หน้าต่างขึ้นมา (Word Cloud, Heat Map ของตารางคำ และ Confusion Matrix)
ถ้าไม่ต้องการเปิดหน้าต่าง ให้ตั้ง `MPLBACKEND=Agg` ก่อนรัน (บันทึกเป็นไฟล์ภาพอย่างเดียว)

> หมายเหตุ: ฟอนต์ภาษาไทยเลือกอัตโนมัติ (Tahoma บน Windows, Thonburi บน macOS, Loma/Garuda บน Linux) หรือกำหนดเองด้วย `WONGNAI_FONT`
