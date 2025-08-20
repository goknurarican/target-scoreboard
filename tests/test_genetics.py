import asyncio
from app.data_access.opentargets import ot_client
from app.channels.genetics import compute_genetics_score

async def _grab():
    data = await ot_client.fetch_ot_association("EFO_0000305","EGFR")
    score, _ = compute_genetics_score("EFO_0000305","EGFR", data)
    return score

def test_egfr_nsclc_genetics_positive():
    score = asyncio.run(_grab())
    assert score >= 0.05  # en azından sıfır değil
