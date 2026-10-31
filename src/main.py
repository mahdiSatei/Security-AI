from processor import AlertProcessor
from vector_store import AlertVectorStore

RAW_INPUT_FILE = "data/security_alerts.json"
STANDARDIZED_OUTPUT_FILE = "data/standardized_alerts.json"


def display_results(title: str, results: list):
    print("\n" + "=" * 70)
    print(f"[*] Query: {title}")
    print("=" * 70)
    if not results:
        print("[-] No matching alerts found.")
        return

    for r in results:
        print(f"Rank #{r['rank']} | ID: {r['alert_id']} | Similarity: {r['similarity_score']:.2%}")
        print(f"   Rule: {r['rule_name']} (Severity: {r['severity'].upper()})")
        print(f"   Source: {r['source_ip']} -> Target: {r['target_resource']}")
        print(f"   Text: {r['text'][:90]}...")
        print("-" * 70)


def main():
 
    processor = AlertProcessor()
    alerts = processor.process_and_save(RAW_INPUT_FILE, STANDARDIZED_OUTPUT_FILE)

    vstore = AlertVectorStore()
    vstore.index_alerts(alerts, force_reindex=False)

    q1 = "database query injection"
    res1 = vstore.search(query=q1, n_results=3)
    display_results(f"'{q1}'", res1)

    q2 = "brute force login password attempts"
    res2 = vstore.search(query=q2, n_results=3, filters={"severity": "high"})
    display_results(f"'{q2}' [Filter: severity=high]", res2)


if __name__ == "__main__":
    main()