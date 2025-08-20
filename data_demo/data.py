#!/usr/bin/env python3
"""
Script to create the required demo data files.
"""
import os
from pathlib import Path


def create_demo_data_files():
    """Create all demo data files."""

    # Ensure data_demo directory exists
    data_dir = Path("data_demo")
    data_dir.mkdir(exist_ok=True)

    # 1. Create expression_demo.csv
    expression_data = """tissue,gene,tpm
lung,EGFR,245.8
lung,CRBN,12.4
lung,VHL,34.2
lung,DDB1,28.9
lung,KRAS,89.3
lung,ERBB2,123.7
lung,MET,67.2
lung,ALK,8.9
lung,BRAF,45.6
breast,EGFR,189.2
breast,CRBN,15.8
breast,VHL,29.1
breast,DDB1,31.5
breast,KRAS,78.9
breast,ERBB2,287.4
breast,MET,52.3
breast,ALK,4.2
breast,BRAF,38.7
liver,EGFR,98.4
liver,CRBN,8.7
liver,VHL,22.6
liver,DDB1,19.3
liver,KRAS,67.8
liver,ERBB2,45.9
liver,MET,156.7
liver,ALK,2.1
liver,BRAF,29.4
brain,EGFR,67.3
brain,CRBN,18.9
brain,VHL,41.2
brain,DDB1,25.7
brain,KRAS,45.2
brain,ERBB2,89.6
brain,MET,78.4
brain,ALK,34.8
brain,BRAF,156.9"""

    with open(data_dir / "expression_demo.csv", "w") as f:
        f.write(expression_data)

    # 2. Create ternary_reports.csv
    ternary_data = """target,evidence_level
EGFR,reported
ERBB2,weak
KRAS,none
MET,weak
ALK,reported
BRAF,weak
PIK3CA,none
TP53,none
BRCA1,none
BRCA2,none
PTEN,none
RB1,none
VHL,reported"""

    with open(data_dir / "ternary_reports.csv", "w") as f:
        f.write(ternary_data)

    # 3. Create hotspot_edges.csv
    hotspot_data = """target,partner,interaction_type,hotspot_score
EGFR,GRB2,binding,0.85
EGFR,SHC1,binding,0.72
EGFR,ERBB2,complex,0.91
ERBB2,GRB2,binding,0.67
ERBB2,EGFR,complex,0.91
KRAS,RAF1,binding,0.78
KRAS,PIK3CA,binding,0.64
MET,GRB2,binding,0.73
MET,GAB1,binding,0.82
ALK,GRB2,binding,0.69
ALK,SHC1,binding,0.75
BRAF,MAP2K1,binding,0.88
BRAF,MAP2K2,binding,0.85
PIK3CA,AKT1,binding,0.76
VHL,HIF1A,complex,0.93
VHL,ELOB,complex,0.89"""

    with open(data_dir / "hotspot_edges.csv", "w") as f:
        f.write(hotspot_data)

    # 4. Update or create ppi_edges.tsv if it doesn't exist
    ppi_file = data_dir / "ppi_edges.tsv"
    if not ppi_file.exists():
        ppi_data = """protein1	protein2	confidence
EGFR	ERBB2	0.95
EGFR	GRB2	0.89
EGFR	SOS1	0.78
ERBB2	GRB2	0.82
KRAS	RAF1	0.92
KRAS	PIK3CA	0.75
RAF1	MAP2K1	0.94
MAP2K1	MAPK1	0.91
PIK3CA	AKT1	0.88
AKT1	MTOR	0.85
TP53	MDM2	0.96
TP53	CDKN1A	0.87
BRCA1	BRCA2	0.79
BRCA1	TP53	0.73
MET	GRB2	0.81
ALK	GRB2	0.76
BRAF	MAP2K1	0.93
PTEN	AKT1	0.84
RB1	E2F1	0.89
VHL	HIF1A	0.97"""

        with open(ppi_file, "w") as f:
            f.write(ppi_data)

    print("Demo data files created successfully:")
    print(f"  - {data_dir}/expression_demo.csv")
    print(f"  - {data_dir}/ternary_reports.csv")
    print(f"  - {data_dir}/hotspot_edges.csv")
    print(f"  - {data_dir}/ppi_edges.tsv")


if __name__ == "__main__":
    create_demo_data_files()