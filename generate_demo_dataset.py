"""
generate_demo_dataset.py
========================
Generates a realistic synthetic phishing/benign URL dataset for training.
Produces ~5,000 URLs with balanced classes.

Usage: python generate_demo_dataset.py --output dataset.csv
"""

import argparse
import random
import string
import csv

random.seed(42)

# --- Benign URL templates ---
BENIGN_DOMAINS = [
    "google.com", "github.com", "stackoverflow.com", "wikipedia.org",
    "amazon.com", "youtube.com", "linkedin.com", "twitter.com",
    "facebook.com", "reddit.com", "microsoft.com", "apple.com",
    "netflix.com", "spotify.com", "dropbox.com", "slack.com",
    "notion.so", "medium.com", "dev.to", "cloudflare.com",
]

BENIGN_PATHS = [
    "/", "/about", "/contact", "/blog/post-1", "/docs/getting-started",
    "/search?q=python", "/products/laptop", "/en/wiki/Python_(programming_language)",
    "/questions/tagged/machine-learning", "/watch?v=dQw4w9WgXcQ",
    "/user/profile", "/news/technology", "/pricing", "/features",
]

# --- Phishing URL templates ---
PHISHING_KEYWORDS = ["login", "verify", "secure", "account", "update", "confirm", "signin"]
PHISHING_BRANDS   = ["paypal", "amazon", "apple", "microsoft", "google", "facebook", "netflix", "ebay"]
PHISHING_TLDS     = [".xyz", ".ru", ".tk", ".ml", ".ga", ".cf", ".gq", ".top", ".click", ".link"]
PHISHING_HOSTS    = [
    "secure-login-verify.{tld}", "{brand}-account-update.{tld}",
    "{brand}-secure.verify-now{tld}", "login-{brand}.account{tld}",
    "update-{brand}-account.com.{tld}", "{brand}.{random}-phishing{tld}",
]
PHISHING_PATHS = [
    "/login/verify", "/account/update", "/confirm/identity",
    "/secure/signin?redirect=true", "/update/credentials",
    "/account/suspended?action=verify", "/secure/payment/confirm",
]


def rand_str(n=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))


def rand_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def generate_benign_url():
    scheme = random.choice(["https://", "http://"])
    domain = random.choice(BENIGN_DOMAINS)
    sub = random.choice(["", "www.", "docs.", "api.", "blog."])
    path = random.choice(BENIGN_PATHS)
    return f"{scheme}{sub}{domain}{path}"


def generate_phishing_url():
    style = random.randint(0, 4)

    if style == 0:
        # Brand + suspicious TLD
        brand  = random.choice(PHISHING_BRANDS)
        kw     = random.choice(PHISHING_KEYWORDS)
        tld    = random.choice(PHISHING_TLDS)
        path   = random.choice(PHISHING_PATHS)
        return f"http://{brand}-{kw}-account.com{tld}{path}"

    elif style == 1:
        # IP address-based
        ip   = rand_ip()
        kw   = random.choice(PHISHING_KEYWORDS)
        path = f"/{kw}/" + rand_str(6) + "?user=admin@" + random.choice(PHISHING_BRANDS) + ".com"
        return f"http://{ip}{path}"

    elif style == 2:
        # Subdomain spoofing
        brand  = random.choice(PHISHING_BRANDS)
        kw     = random.choice(PHISHING_KEYWORDS)
        random_domain = rand_str(8) + random.choice(PHISHING_TLDS)
        path   = random.choice(PHISHING_PATHS)
        return f"http://{brand}.{kw}.{brand}.{random_domain}{path}"

    elif style == 3:
        # @ redirect trick
        brand  = random.choice(PHISHING_BRANDS)
        real   = brand + ".com"
        evil   = rand_str(10) + random.choice(PHISHING_TLDS)
        return f"http://{real}@{evil}/{random.choice(PHISHING_KEYWORDS)}"

    else:
        # Long obfuscated URL with % encoding
        brand  = random.choice(PHISHING_BRANDS)
        kw     = random.choice(PHISHING_KEYWORDS)
        tld    = random.choice(PHISHING_TLDS)
        noise  = "-".join(rand_str(4) for _ in range(4))
        params = "?user=%61%64%6d%69%6e&token=" + rand_str(24)
        return f"http://{brand}-{kw}-{noise}{tld}/update/credentials{params}"


def generate_dataset(n_each: int) -> list:
    rows = []
    for _ in range(n_each):
        rows.append({"url": generate_benign_url(),   "label": 0})
        rows.append({"url": generate_phishing_url(), "label": 1})
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dataset.csv")
    parser.add_argument("--n", type=int, default=2500, help="URLs per class")
    args = parser.parse_args()

    data = generate_dataset(args.n)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label"])
        writer.writeheader()
        writer.writerows(data)

    print(f"✅ Dataset generated: {args.output} ({len(data)} rows, balanced classes)")
