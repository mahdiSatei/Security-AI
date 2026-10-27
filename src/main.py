from processor import AlertProcessor

INPUT_FILE = "../data/security_alerts.json"
OUTPUT_FILE = "../data/standardized_alerts.json"

if __name__ == "__main__":
    processor = AlertProcessor()
    alerts = processor.process_and_save(INPUT_FILE, OUTPUT_FILE)

    print("\n--- Sample Processed Alert ---")
    print(f"ID: {alerts[0]['alert_id']}")
    print(f"Embedding Text: {alerts[0]['embedding_text']}")