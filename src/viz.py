import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("reports/figures", exist_ok=True)
df = pd.read_csv("data/raw/student_performance.csv")
plt.figure()
df['exam_score'].hist(bins=20)
plt.title('Exam Score Distribution')
plt.xlabel('Score')
plt.ylabel('Count')
plt.savefig("reports/figures/exam_score_hist.png", bbox_inches='tight')
print("Saved reports/figures/exam_score_hist.png")
