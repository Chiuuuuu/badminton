import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

MAX_TIME_DIFF = 5  # 允許的時間誤差（秒）

def evaluate(predicted_df, label_df):
    evaluate_results_list = []
    TP, FP, FN = 0, 0, 0  # 這裡不計算 TN

    # 遍歷 label_df
    for row in label_df.itertuples():
        label_time = row.Time
        label_scorer = row.Scorer
        error_type = None

        # 找到與 label_time 最接近的 predicted_df 之 row
        closest_row = predicted_df.iloc[(predicted_df['Time'] - label_time).dt.total_seconds().abs().argsort()[:1]]

        if not closest_row.empty:
            closest_row_time_diff = abs((closest_row['Time'].values[0] - label_time).total_seconds())
            predicted_scorer = closest_row['Scorer'].values[0]

            # 判斷預測是否正確
            founded = closest_row_time_diff <= MAX_TIME_DIFF
            correct = int(founded and predicted_scorer == label_scorer)

            if not founded:
                correct = 0

            predicted_time = closest_row['Time'].values[0]

            if founded and not correct:
                error_type = 'Wrong_Scorer'
                FP += 1  # 記錯 Scorer 算 False Positive

            elif not founded:
                predicted_scorer = None
                predicted_time = None
                error_type = 'Missed_Score'
                FN += 1  # Missed Score 算 False Negative

            if correct:
                TP += 1  # 預測正確的得分

            evaluate_results_list.append({
                'Predicted_Time': pd.Timestamp(predicted_time).strftime('%H:%M:%S') if founded else None,
                'Predicted_Scorer': predicted_scorer,
                'Label_Time': pd.Timestamp(label_time).strftime('%H:%M:%S'),
                'Label_Scorer': label_scorer,
                'Correct': correct,
                'Error_Type': error_type
            })

    # 遍歷 predicted_df，找沒有對應的 label（False Trigger）
    for row in predicted_df.itertuples():
        predicted_time = row.Time
        predicted_scorer = row.Scorer

        # 找到與 predicted_time 最接近的 label_df 之 row
        closest_row = label_df.iloc[(label_df['Time'] - predicted_time).dt.total_seconds().abs().argsort()[:1]]

        if not closest_row.empty:
            closest_row_time_diff = abs((closest_row['Time'].values[0] - predicted_time).total_seconds())
            label_scorer = closest_row['Scorer'].values[0]

            # 判斷預測是否正確
            founded = closest_row_time_diff <= MAX_TIME_DIFF
            correct = int(founded and predicted_scorer == label_scorer)
            if not founded:
                correct = 0
        else:
            # 如果找不到對應的 label，則為 False Trigger
            correct = 0
            label_scorer = None

        if not founded:
            evaluate_results_list.append({
                'Predicted_Time': pd.Timestamp(predicted_time).strftime('%H:%M:%S'),
                'Predicted_Scorer': predicted_scorer,
                'Label_Time': None,
                'Label_Scorer': None,
                'Correct': correct,
                'Error_Type': 'False_Trigger'
            })
            FP += 1  # False Trigger 算 False Positive

    # 計算評估指標
    accuracy = TP / (TP + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")

    # 轉為 DataFrame
    evaluate_results_df = pd.DataFrame(evaluate_results_list)

    evaluate_results_df['Predicted_Scorer'] = pd.to_numeric(evaluate_results_df['Predicted_Scorer'], errors='coerce').astype('Int64')
    evaluate_results_df['Label_Scorer'] = pd.to_numeric(evaluate_results_df['Label_Scorer'], errors='coerce').astype('Int64')

    # 由時間排序
    evaluate_results_df = evaluate_results_df.sort_values(by='Predicted_Time').reset_index(drop=True)
    
    return evaluate_results_df, accuracy, precision, recall, f1_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted', type=str, default='scored.csv', help='CSV with predicted scores')
    parser.add_argument('--label', type=str, default='label.csv', help='CSV with ground truth labels')
    parser.add_argument('--output_csv', type=str, default='evaluate_results/evaluation.csv', help='Path to save evaluation result CSV')
    parser.add_argument('--output_txt', type=str, default='evaluation_metrics/metrics.txt', help='Path to save metrics text file')
    args = parser.parse_args()

    # 讀取 CSV 並確保時間格式正確
    predicted_df = pd.read_csv(args.predicted)
    predicted_df['Time'] = pd.to_datetime(predicted_df['Time'])

    label_df = pd.read_csv(args.label)
    label_df['Time'] = pd.to_datetime(label_df['Time'])

    # 執行評估
    evaluate_results_df, accuracy, precision, recall, f1_score = evaluate(predicted_df, label_df)

    # 儲存結果
    evaluate_results_df.to_csv(args.output_csv, index=False)

    # 儲存評估指標
    with open(args.output_txt, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1_score:.4f}\n")

    print("評估完成，結果已儲存至 evaluate_results 和 evaluation_metrics")
