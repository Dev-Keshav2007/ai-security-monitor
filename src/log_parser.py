def read_logs(file_path):
    with open(file_path, "r") as file:
        logs = file.readlines()

    return logs


def parse_log(log):
    parts = log.strip().split()

    timestamp = parts[0] + " " + parts[1]
    event = parts[2]
    user = parts[3].split("=")[1]
    ip = parts[4].split("=")[1]

    return {
        "timestamp": timestamp,
        "event": event,
        "user": user,
        "ip": ip
    }