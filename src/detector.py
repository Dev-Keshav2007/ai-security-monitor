from datetime import datetime


def detect_brute_force(events):
    failed_logins = {}

    for event in events:
        if event["event"] == "LOGIN_FAILED":
            ip = event["ip"]

            timestamp = datetime.strptime(
                event["timestamp"],
                "%Y-%m-%d %H:%M:%S"
            )

            if ip not in failed_logins:
                failed_logins[ip] = []

            failed_logins[ip].append(timestamp)

    alerts = []

    for ip, timestamps in failed_logins.items():
        timestamps.sort()

        for i in range(len(timestamps)):
            attempts = 1

            for j in range(i + 1, len(timestamps)):
                difference = timestamps[j] - timestamps[i]

                if difference.total_seconds() <= 120:
                    attempts += 1
                else:
                    break

            if attempts >= 5:
                alerts.append({
                    "type": "BRUTE_FORCE",
                    "ip": ip,
                    "failed_attempts": attempts,
                    "severity": "HIGH",
                    "time_window": "2 minutes"
                })
                break

    return alerts


def detect_account_spray(events):
    ip_users = {}

    for event in events:
        if event["event"] == "LOGIN_FAILED":
            ip = event["ip"]
            user = event["user"]

            if ip not in ip_users:
                ip_users[ip] = set()

            ip_users[ip].add(user)

    alerts = []

    for ip, users in ip_users.items():
        if len(users) >= 4:
            alerts.append({
                "type": "ACCOUNT_SPRAY",
                "ip": ip,
                "accounts_targeted": len(users),
                "severity": "MEDIUM"
            })

    return alerts