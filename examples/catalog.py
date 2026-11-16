"""An illustrative mid-market control catalog and loss scenarios.

IMPORTANT: these frequencies, magnitudes and control efficacies are *illustrative*
defaults, chosen to be plausible for a ~500-person company, so the method is runnable
end to end. They are the inputs a CISO calibrates to their own environment (loss data,
threat intel, control assessments). The method is the asset; the numbers are a worked
example, not a benchmark.
"""
from riskopt.model import Scenario, Control

SCENARIOS = [
    Scenario("ransomware", "Ransomware / extortion",        lef=0.40, lm_mean=1_800_000, lm_cv=1.8),
    Scenario("bec",        "Business email compromise",     lef=0.90, lm_mean=120_000,   lm_cv=1.4),
    Scenario("datatheft",  "Insider / credential data theft", lef=0.30, lm_mean=900_000, lm_cv=1.6),
    Scenario("cloud",      "Cloud misconfiguration breach", lef=0.50, lm_mean=650_000,   lm_cv=1.7),
    Scenario("webapp",     "Web-app exploit / data breach", lef=0.60, lm_mean=500_000,   lm_cv=1.6),
    Scenario("ddos",       "DDoS / availability loss",      lef=1.20, lm_mean=60_000,    lm_cv=1.2),
]

# freq_reduction / mag_reduction are fractions in [0,1) applied to the named scenarios.
CONTROLS = [
    Control("mfa", "Phishing-resistant MFA + SSO", 90_000,
            freq_reduction={"bec": 0.70, "datatheft": 0.45, "cloud": 0.25, "ransomware": 0.30}),
    Control("edr", "EDR / XDR on endpoints", 140_000,
            freq_reduction={"ransomware": 0.45, "datatheft": 0.30},
            mag_reduction={"ransomware": 0.25}),
    Control("email", "Advanced email filtering", 45_000,
            freq_reduction={"bec": 0.55, "ransomware": 0.25}),
    Control("backups", "Immutable, tested backups", 110_000,
            mag_reduction={"ransomware": 0.65}),
    Control("dlp", "DLP + egress monitoring", 80_000,
            freq_reduction={"datatheft": 0.40}, mag_reduction={"datatheft": 0.30}),
    Control("netseg", "Network segmentation", 130_000,
            freq_reduction={"ransomware": 0.30}, mag_reduction={"ransomware": 0.35, "datatheft": 0.25}),
    Control("waf", "WAF + managed rules", 55_000,
            freq_reduction={"webapp": 0.60, "ddos": 0.50}),
    Control("casb", "Cloud posture mgmt (CSPM/CASB)", 95_000,
            freq_reduction={"cloud": 0.65}, mag_reduction={"cloud": 0.20}),
    Control("siem", "SIEM + 24x7 detection", 220_000,
            freq_reduction={"datatheft": 0.20, "cloud": 0.20, "webapp": 0.20},
            mag_reduction={"ransomware": 0.20, "datatheft": 0.25, "cloud": 0.25, "webapp": 0.25}),
    Control("training", "Security-awareness training", 30_000,
            freq_reduction={"bec": 0.35, "ransomware": 0.15}),
    Control("patch", "Patch & vuln management", 85_000,
            freq_reduction={"webapp": 0.45, "ransomware": 0.20, "cloud": 0.15}),
    Control("pam", "Privileged access management", 120_000,
            freq_reduction={"datatheft": 0.35, "cloud": 0.25},
            mag_reduction={"datatheft": 0.20}),
    Control("pentest", "Annual pentest + remediation", 60_000,
            freq_reduction={"webapp": 0.30, "cloud": 0.20}),
    Control("ir", "IR retainer + playbooks", 70_000,
            mag_reduction={"ransomware": 0.20, "datatheft": 0.15, "cloud": 0.15, "webapp": 0.15, "bec": 0.15}),
]


def build():
    from riskopt.model import RiskModel
    return RiskModel(SCENARIOS, CONTROLS)
