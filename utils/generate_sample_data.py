#!/usr/bin/env python3
"""
Generate realistic synthetic data for testing the Business Entity Resolution pipeline.
Generates:
- dataset/sample/train: train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
- dataset/sample/test: test_source1.tsv, test_source2.tsv, test_source3.tsv
Includes US, India, and France (test-only) entities with real-world noise patterns.
"""

import os
import random
import csv

random.seed(42)

US_ENTITIES = [
    ("Apple Inc.", "1 Infinite Loop, Cupertino, CA 95014", "US"),
    ("Microsoft Corporation", "One Microsoft Way, Redmond, WA 98052", "US"),
    ("Amazon.com Services LLC", "410 Terry Ave N, Seattle, WA 98109", "US"),
    ("Google LLC", "1600 Amphitheatre Pkwy, Mountain View, CA 94043", "US"),
    ("Meta Platforms Inc.", "1 Hacker Way, Menlo Park, CA 94025", "US"),
    ("Tesla Motors Inc.", "3500 Deer Creek Rd, Palo Alto, CA 94304", "US"),
    ("Target Corporation", "1000 Nicollet Mall, Minneapolis, MN 55403", "US"),
    ("Costco Wholesale Corp", "999 Lake Dr, Issaquah, WA 98027", "US"),
    ("Starbucks Coffee Company", "2401 Utah Ave S, Seattle, WA 98134", "US"),
    ("The Home Depot Inc.", "2455 Paces Ferry Rd NW, Atlanta, GA 30339", "US"),
    ("Johnson & Johnson", "One Johnson & Johnson Plaza, New Brunswick, NJ 08933", "US"),
    ("Walmart Inc.", "702 SW 8th St, Bentonville, AR 72716", "US"),
    ("Pfizer Inc.", "235 E 42nd St, New York, NY 10017", "US"),
    ("CVS Health Corp", "One CVS Dr, Woonsocket, RI 02895", "US"),
    ("JPMorgan Chase & Co.", "383 Madison Ave, New York, NY 10179", "US"),
]

INDIA_ENTITIES = [
    ("Tata Consultancy Services Ltd", "Nirmal Building, 9th Floor, Nariman Point, Mumbai 400021", "India"),
    ("Infosys Limited", "Electronics City, Hosur Road, Bengaluru 560100", "India"),
    ("Reliance Industries Limited", "Maker Chambers IV, 3rd Floor, 222 Nariman Point, Mumbai 400021", "India"),
    ("HDFC Bank Ltd", "HDFC Bank House, Senapati Bapat Marg, Lower Parel, Mumbai 400013", "India"),
    ("Wipro Enterprises Pvt Ltd", "Doddakannelli, Sarjapur Road, Bengaluru 560035", "India"),
    ("State Bank of India", "State Bank Bhavan, Madame Cama Road, Nariman Point, Mumbai 400021", "India"),
    ("Bharti Airtel Limited", "Bharti Crescent, 1 Nelson Mandela Road, Vasant Kunj, New Delhi 110070", "India"),
    ("Mahindra & Mahindra Ltd", "Gateway Building, Apollo Bunder, Mumbai 400039", "India"),
    ("ITC Limited", "Virginia House, 37 J.L. Nehru Road, Kolkata 700071", "India"),
    ("Larsen & Toubro Ltd", "L&T House, Ballard Estate, Mumbai 400001", "India"),
    ("Bajaj Finance Ltd", "Akurdi, Pune 411035", "India"),
    ("Zomato Limited", "Ground Floor, 12A, 94 Meghdoot, Nehru Place, New Delhi 110019", "India"),
    ("Swiggy Private Limited", "IBC Knowledge Park, Bannerghatta Main Rd, Bengaluru 560029", "India"),
]

FRANCE_ENTITIES = [
    ("L'Oreal SA", "41 Rue Martre, 92117 Clichy", "France"),
    ("TotalEnergies SE", "2 Place Jean Millier, La Defense 6, 92400 Courbevoie", "France"),
    ("LVMH Moet Hennessy Louis Vuitton", "22 Avenue Montaigne, 75008 Paris", "France"),
    ("Sanofi SA", "46 Avenue de la Grande Armee, 75017 Paris", "France"),
    ("BNP Paribas SA", "16 Boulevard des Italiens, 75009 Paris", "France"),
    ("Danone SA", "17 Boulevard Haussmann, 75009 Paris", "France"),
    ("Renault Group", "122-122 bis Avenue du General Leclerc, 92100 Boulogne-Billancourt", "France"),
    ("Airbus SE France", "1 Rond-Point Maurice Bellonte, 31707 Blagnac", "France"),
]


def corrupt_name(name):
    transforms = [
        lambda s: s.replace("Corporation", "Corp").replace("Company", "Co"),
        lambda s: s.replace("Limited", "Ltd").replace("Private Limited", "Pvt Ltd"),
        lambda s: s.replace("&", "and"),
        lambda s: s.replace("and", "&"),
        lambda s: s.replace("LLC", "").replace("Inc.", "").replace("SA", "").strip(),
        lambda s: s.lower(),
        lambda s: s.upper(),
        lambda s: s + " Store #102",
        lambda s: "The " + s if not s.startswith("The ") else s[4:],
    ]
    t = random.choice(transforms)
    new_name = t(name)
    # 20% chance of small typo
    if random.random() < 0.2 and len(new_name) > 4:
        idx = random.randint(1, len(new_name) - 2)
        new_name = new_name[:idx] + new_name[idx + 1] + new_name[idx] + new_name[idx + 2 :]
    return new_name


def corrupt_address(addr):
    transforms = [
        lambda s: s.replace("Street", "St").replace("Road", "Rd").replace("Avenue", "Ave"),
        lambda s: s.replace("Boulevard", "Blvd").replace("Floor", "Fl"),
        lambda s: "Near " + s.split(",")[0] + ", " + ", ".join(s.split(",")[1:]),
        lambda s: s + " (Opposite Bank)",
        lambda s: ", ".join(reversed(s.split(", "))),
        lambda s: "".join([c for c in s if not c.isdigit()]).strip(", "),  # missing PIN/zip
        lambda s: s.split(",")[0] + ", " + s.split(",")[-1],  # partial address
    ]
    t = random.choice(transforms)
    return t(addr)


def generate_dataset_split(base_entities, output_dir, is_train=True, prefix="train"):
    os.makedirs(output_dir, exist_ok=True)
    s1_rows = []
    s2_rows = []
    s3_rows = []
    ground_truth = []

    s1_counter = 1
    s2_counter = 1
    s3_counter = 1

    for name, addr, country in base_entities:
        s1_id = f"S1-{s1_counter:05d}"
        s1_counter += 1
        s1_rows.append((s1_id, name, addr, country))

        # Determine matches in S2 and S3
        match_type = random.choice(["both", "s2_only", "s3_only", "multiple_s2", "none"])
        matched_ids = []

        if match_type in ["both", "s2_only", "multiple_s2"]:
            s2_id = f"S2-{s2_counter:05d}"
            s2_counter += 1
            s2_rows.append((s2_id, corrupt_name(name), corrupt_address(addr), country))
            matched_ids.append(s2_id)

            if match_type == "multiple_s2":
                s2_id2 = f"S2-{s2_counter:05d}"
                s2_counter += 1
                s2_rows.append((s2_id2, corrupt_name(name), corrupt_address(addr), country))
                matched_ids.append(s2_id2)

        if match_type in ["both", "s3_only"]:
            s3_id = f"S3-{s3_counter:05d}"
            s3_counter += 1
            s3_rows.append((s3_id, corrupt_name(name), corrupt_address(addr), country))
            matched_ids.append(s3_id)

        # Distractor entities in S2/S3 (false positives)
        if random.random() < 0.25:
            distractor_s2_id = f"S2-{s2_counter:05d}"
            s2_counter += 1
            s2_rows.append(
                (
                    distractor_s2_id,
                    corrupt_name(name) + " Trading",
                    corrupt_address(addr).replace("95014", "90210").replace("400021", "110001"),
                    country,
                )
            )

        if is_train:
            ground_truth.append((s1_id, ",".join(matched_ids)))

    # Save to TSV files
    def write_tsv(filename, header, data):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(header)
            writer.writerows(data)
        print(f"Generated {filepath} ({len(data)} rows)")

    header_source = ["entity_id", "business_name", "business_address", "country"]
    write_tsv(f"{prefix}_source1.tsv", header_source, s1_rows)
    write_tsv(f"{prefix}_source2.tsv", header_source, s2_rows)
    write_tsv(f"{prefix}_source3.tsv", header_source, s3_rows)

    if is_train:
        write_tsv(
            f"{prefix}_ground_truth.tsv",
            ["source1_entity_id", "matched_entity_ids"],
            ground_truth,
        )


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "sample")

    train_entities = US_ENTITIES[:10] + INDIA_ENTITIES[:10]
    test_entities = US_ENTITIES[10:] + INDIA_ENTITIES[10:] + FRANCE_ENTITIES

    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    print("Generating sample training dataset (US, India)...")
    generate_dataset_split(train_entities, train_dir, is_train=True, prefix="train")

    print("\nGenerating sample test dataset (US, India, France)...")
    generate_dataset_split(test_entities, test_dir, is_train=False, prefix="test")

    print("\nSample dataset generation completed successfully.")


if __name__ == "__main__":
    main()
