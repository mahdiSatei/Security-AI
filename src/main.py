import json
from processor import AlertProcessor
from vector_store import AlertVectorStore

RAW_INPUT_FILE = "data/security_alerts.json"
STANDARDIZED_OUTPUT_FILE = "data/standardized_alerts.json"


def display_results(title: str, results: list):
    print("\n" + "=" * 65)
    print(f"🔍 {title}")
    print("=" * 65)
    if not results:
        print("موردی یافت نشد.")
        return

    for r in results:
        print(f"رتبه #{r['rank']} | شناسه: {r['alert_id']} | شباهت: {r['similarity_score']:.2%}")
        print(f"   عنوان حمله: {r['rule_name']} (سطح: {r['severity'].upper()})")
        print(f"   منبع: {r['source_ip']} -> منبع هدف: {r['target_resource']}")
        print(f"   متن ایندکس‌شده: {r['text'][:90]}...")
        print("-" * 65)


def main():
    # ۱. اجرای بخش اول: پردازش و استانداردسازی
    processor = AlertProcessor()
    alerts = processor.process_and_save(RAW_INPUT_FILE, STANDARDIZED_OUTPUT_FILE)

    # ۲. اجرای بخش دوم: بردارسازی و ذخیره در پایگاه‌داده ChromaDB
    vstore = AlertVectorStore()
    vstore.index_alerts(alerts)

    # ۳. تست جستجوی معنایی (بدون ذکر نام مستقیم SQL)
    q1 = "database manipulation and query injection"
    res1 = vstore.search(query=q1, n_results=3)
    display_results(f"جستجو: '{q1}'", res1)

    # ۴. تست جستجوی معنایی با فیلتر شدت (فقط هشدارهای high)
    q2 = "brute force login password attempts"
    res2 = vstore.search(query=q2, n_results=3, severity_filter="high")
    display_results(f"جستجو: '{q2}' (فیلتر: Severity=high)", res2)


if __name__ == "__main__":
    main()