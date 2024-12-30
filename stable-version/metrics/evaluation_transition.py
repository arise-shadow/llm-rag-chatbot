from metrics.quality_metrics import calculate_bleu_meteor_score, calculate_bert_score
from time import time

def evaluate_translation(result: str, ref_text: str, elapsed_time: float, metrics_config: dict):
    """
    Evaluates translation quality and performance based on the selected metrics.

    Parameters:
        result (str): Translated text.
        ref_text (str): Reference text for evaluation.
        elapsed_time (float): Time taken for translation.
        metrics_config (dict): Configuration for enabled metrics (e.g., {"BLEU": True, "METEOR": False}).

    Returns:
        dict: A dictionary containing only the enabled metrics and their values.
    """
    metrics = {}

    # Calculate BLEU and METEOR if enabled
    if metrics_config.get("BLEU", False) or metrics_config.get("METEOR", False):
        try:
            bleu, meteor, num_tokens = calculate_bleu_meteor_score(ref_text, result)
            if metrics_config.get("BLEU", False):
                metrics["BLEU"] = bleu
            if metrics_config.get("METEOR", False):
                metrics["METEOR"] = meteor
            if metrics_config.get("TPS", False):
                metrics["TPS"] = num_tokens / elapsed_time if elapsed_time > 0 else 0.0
        except Exception as e:
            print(f"Error calculating BLEU/METEOR: {e}")
            if metrics_config.get("BLEU", False):
                metrics["BLEU"] = 0.0
            if metrics_config.get("METEOR", False):
                metrics["METEOR"] = 0.0
            if metrics_config.get("TPS", False):
                metrics["TPS"] = 0.0

    # Calculate BERTScore if enabled
    if metrics_config.get("BERTScore", False):
        try:
            bert_score = calculate_bert_score(ref_text, result, tgt_lang="English", device="cuda")
            metrics["BERTScore"] = bert_score
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            metrics["BERTScore"] = 0.0

    return metrics