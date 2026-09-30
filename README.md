# AI Security Monitor

A Python-based cybersecurity monitoring project that analyzes authentication logs and detects suspicious login activity.

## Current Features

- Parses raw security log data into structured events
- Detects repeated failed login attempts
- Detects brute-force attacks within a defined time window
- Detects account-spray activity across multiple usernames
- Assigns severity levels to detected threats
- Displays security alerts with source IP and attack information

## Detection Rules

### Brute-Force Detection
The system generates a HIGH severity alert when an IP address produces 5 or more failed login attempts within 2 minutes.

### Account-Spray Detection
The system generates a MEDIUM severity alert when one IP address attempts to access 4 or more different user accounts.

## Project Structure

    ai-security-monitor/
    ├── data/
    │   └── security.log
    ├── src/
    │   ├── main.py
    │   ├── log_parser.py
    │   └── detector.py
    ├── tests/
    ├── .gitignore
    ├── README.md
    └── requirements.txt

## Technologies

- Python
- Git
- GitHub
- Visual Studio Code
- Security log analysis
- Rule-based threat detection

## Run the Project

From the project directory:

    python3 src/main.py

## Example Detection

    SECURITY ALERTS
    ================

    Threat: BRUTE_FORCE
    Source IP: 45.22.18.91
    Failed Attempts: 5
    Time Window: 2 minutes
    Severity: HIGH

    Threat: ACCOUNT_SPRAY
    Source IP: 72.44.18.50
    Accounts Targeted: 4
    Severity: MEDIUM

## Roadmap

This project is being developed toward a more advanced AI-assisted security monitoring system. Planned improvements include:

- Additional threat detection rules
- Automated testing
- Security event scoring
- Machine-learning-based anomaly detection
- AI-generated threat explanations
- Security dashboard and visualization
- API integration