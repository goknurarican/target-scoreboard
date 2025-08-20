"""
ID mapping utilities for gene symbols and identifiers.
"""
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Common gene symbol to ENSG mappings for quick lookup
# This is a fallback for when Open Targets API is unavailable
GENE_SYMBOL_TO_ENSG = {
    # NSCLC targets
    "EGFR": "ENSG00000146648",
    "ERBB2": "ENSG00000141736",
    "MET": "ENSG00000105976",
    "ALK": "ENSG00000171094",
    "KRAS": "ENSG00000133703",
    "BRAF": "ENSG00000157764",
    "PIK3CA": "ENSG00000121879",
    "ROS1": "ENSG00000047936",
    "RET": "ENSG00000165731",
    "NTRK1": "ENSG00000198400",

    # Tumor suppressors
    "TP53": "ENSG00000141510",
    "RB1": "ENSG00000139687",
    "PTEN": "ENSG00000171862",

    # Other common targets
    "BRCA1": "ENSG00000012048",
    "BRCA2": "ENSG00000139618",
    "APC": "ENSG00000134982",
    "VHL": "ENSG00000134086",
    "ATM": "ENSG00000149311",
    "CDKN2A": "ENSG00000147889",
    "MLH1": "ENSG00000076242",
    "MSH2": "ENSG00000095002",
    "PALB2": "ENSG00000083093",
    "RAD51": "ENSG00000051180"
}

# Reverse mapping
ENSG_TO_GENE_SYMBOL = {v: k for k, v in GENE_SYMBOL_TO_ENSG.items()}

def normalize_gene_symbol(symbol: str) -> str:
    """
    Normalize gene symbol to standard format.

    Args:
        symbol: Gene symbol in any case

    Returns:
        Normalized gene symbol (uppercase, trimmed)
    """
    if not symbol:
        return ""

    return symbol.strip().upper()

def get_ensg_from_symbol(symbol: str) -> Optional[str]:
    """
    Get ENSG ID from gene symbol using local mapping.

    Args:
        symbol: Gene symbol like "EGFR"

    Returns:
        ENSG ID or None if not found
    """
    normalized_symbol = normalize_gene_symbol(symbol)
    return GENE_SYMBOL_TO_ENSG.get(normalized_symbol)

def get_symbol_from_ensg(ensg_id: str) -> Optional[str]:
    """
    Get gene symbol from ENSG ID using local mapping.

    Args:
        ensg_id: ENSG identifier like "ENSG00000146648"

    Returns:
        Gene symbol or None if not found
    """
    return ENSG_TO_GENE_SYMBOL.get(ensg_id)

def is_ensg_id(identifier: str) -> bool:
    """
    Check if identifier is an ENSG ID.

    Args:
        identifier: String to check

    Returns:
        True if identifier looks like ENSG ID
    """
    return identifier.startswith("ENSG") and len(identifier) >= 15

def validate_gene_identifier(identifier: str) -> tuple[bool, str, str]:
    """
    Validate and normalize gene identifier.

    Args:
        identifier: Gene symbol or ENSG ID

    Returns:
        (is_valid, normalized_id, id_type)
        id_type is either "symbol", "ensg", or "unknown"
    """
    if not identifier:
        return False, "", "unknown"

    identifier = identifier.strip()

    if is_ensg_id(identifier):
        return True, identifier, "ensg"

    # Try as gene symbol
    normalized_symbol = normalize_gene_symbol(identifier)
    if len(normalized_symbol) >= 2:  # Minimum gene symbol length
        return True, normalized_symbol, "symbol"

    return False, identifier, "unknown"

def get_common_cancer_genes() -> Dict[str, str]:
    """
    Get mapping of common cancer genes to their ENSG IDs.

    Returns:
        Dict mapping gene symbols to ENSG IDs
    """
    return GENE_SYMBOL_TO_ENSG.copy()

def add_gene_mapping(symbol: str, ensg_id: str) -> bool:
    """
    Add new gene symbol to ENSG mapping.

    Args:
        symbol: Gene symbol
        ensg_id: ENSG identifier

    Returns:
        True if added successfully
    """
    if not symbol or not ensg_id:
        return False

    if not is_ensg_id(ensg_id):
        logger.warning(f"Invalid ENSG ID format: {ensg_id}")
        return False

    normalized_symbol = normalize_gene_symbol(symbol)
    GENE_SYMBOL_TO_ENSG[normalized_symbol] = ensg_id
    ENSG_TO_GENE_SYMBOL[ensg_id] = normalized_symbol

    logger.info(f"Added gene mapping: {normalized_symbol} -> {ensg_id}")
    return True