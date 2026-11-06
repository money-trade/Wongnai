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
6. **เทรน Naive Bayes** แบ่ง train/test 80/20 (stratified) แล้วแสดง Test Accuracy และ Classification Report
7. **Heat Map** ของ Confusion Matrix อีก 1 หน้าต่าง
8. **Heat Map ของตารางคำ** P(word | star) สำหรับ 30 คำยอดนิยม อีก 1 หน้าต่าง

## ผลลัพธ์

- Test Accuracy ≈ **0.51**
- คลาสไม่สมดุล (3 และ 4 ดาวรวมกันเกิน 75%) โมเดลจึงทำนาย 1 ดาวได้แย่มาก

## ไฟล์ในโปรเจกต์

| ไฟล์ | คำอธิบาย |
|---|---|
| `wongnai_nb.py` | สคริปต์หลัก ทำทุกขั้นตอนข้างต้น |
| `word_table_by_star.csv` | ตารางคำที่สร้างได้ |
| `wordcloud_by_star.png` | ภาพ Word Cloud แยกตามดาว |
| `word_heatmap.png` | Heat Map ของ P(word \| star) สำหรับ 30 คำยอดนิยม |
| `confusion_matrix.png` | ภาพ Confusion Matrix |
| `review_dataset/` | ชุดข้อมูล test และ sample submission |

## วิธีรัน

ติดตั้งไลบรารี:

```bash
pip install pythainlp pandas scikit-learn matplotlib seaborn wordcloud
```

แก้ path ของไฟล์ train (`TRAIN`) ที่ต้นไฟล์ `wongnai_nb.py` ให้ตรงกับเครื่อง แล้วรัน (ไฟล์ผลลัพธ์จะถูกบันทึกในโฟลเดอร์เดียวกับสคริปต์):

```bash
python wongnai_nb.py
```

ใช้เวลาตัดคำประมาณ 1–2 นาที แล้วจะมี 3 หน้าต่างขึ้นมา (Word Cloud, Heat Map ของตารางคำ และ Confusion Matrix)
ถ้าไม่ต้องการเปิดหน้าต่าง ให้ตั้ง `MPLBACKEND=Agg` ก่อนรัน (บันทึกเป็นไฟล์ภาพอย่างเดียว)

> หมายเหตุ: ฟอนต์ Word Cloud ใช้ `C:\Windows\Fonts\tahoma.ttf` (Windows) ถ้าใช้ระบบอื่นต้องเปลี่ยน `FONT` เป็นฟอนต์ที่รองรับภาษาไทย
