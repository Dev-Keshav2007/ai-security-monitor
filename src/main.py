from log_parser import read_logs, parse_log
from detector import detect_brute_force, detect_account_spray


def main():
    print("AI Security Monitor starting...\n")

    logs = read_logs("data/security.log")

    events = []

    for log in logs:
        event = parse_log(log)
        events.append(event)

    alerts = detect_brute_force(events)
    alerts += detect_account_spray(events)

    print("Security scan complete.\n")

    if alerts:
        print("SECURITY ALERTS")
        print("================")

        for alert in alerts:
            print("Threat:", alert["type"])
            print("Source IP:", alert["ip"])

            if alert["type"] == "BRUTE_FORCE":
                print("Failed Attempts:", alert["failed_attempts"])
                print("Time Window:", alert["time_window"])

            elif alert["type"] == "ACCOUNT_SPRAY":
                print("Accounts Targeted:", alert["accounts_targeted"])

            print("Severity:", alert["severity"])
            print("----------------")

    else:
        print("No threats detected.")


if __name__ == "__main__":
    main()