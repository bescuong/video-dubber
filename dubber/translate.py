"""Offline machine translation with Meta NLLB-200."""
import gc
import warnings


def translate(texts, src_nllb, tgt_nllb, model_name, use_gpu=True, log=print, progress=None,
              batch_size=8):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    if not texts:
        return []

    tok = AutoTokenizer.from_pretrained(model_name, src_lang=src_nllb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()

    device = "cpu"
    if use_gpu and torch.cuda.is_available():
        try:
            model.to("cuda")
            device = "cuda"
        except RuntimeError:
            model.to("cpu")
            torch.cuda.empty_cache()
    if device == "cpu":
        torch.set_num_threads(max(4, torch.get_num_threads()))
    log(f"Dịch bằng {model_name.split('/')[-1]} trên {device.upper()}")

    bos = tok.convert_tokens_to_ids(tgt_nllb)

    def _gen(batch, dev):
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=400).to(dev)
        max_new = int(enc["input_ids"].shape[1] * 2.5) + 20
        with torch.no_grad():
            out = model.generate(**enc, forced_bos_token_id=bos, num_beams=4,
                                 max_new_tokens=max_new)
        return tok.batch_decode(out, skip_special_tokens=True)

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            results.extend(_gen(batch, device))
        except RuntimeError as e:
            if device == "cuda" and "out of memory" in str(e).lower():
                log("GPU hết bộ nhớ khi dịch, chuyển sang CPU")
                model.to("cpu")
                torch.cuda.empty_cache()
                device = "cpu"
                results.extend(_gen(batch, device))
            else:
                raise
        if progress:
            progress(min(1.0, (i + len(batch)) / len(texts)))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [r.strip() for r in results]
