# 1) Build your pretrained vocab (your class)
#vocab = PretrainedGeometricVocab(
#    repo_id="AbstractPhil/geometric-vocab-32d",
#    dim=32,
#    split="wordnet_eng",
#    store="both",             # in your PretrainedGeometricVocab
#    manifest_specials=True,
#)
#
## 2) Create the device bank
#bank = VocabDeviceBank.from_pretrained(
#    vocab,
#    name="geovocab-wordnet-eng",
#    store="full",             # "pooled" if you only need pooled
#    finalize_mode="post_mean",
#    normalize="l1",
#    pinned_cpu=True,
#    cache_pooled=True,
#)
#VocabBankRegistry.get_or_register(bank)
#
## 3) Share CPU (if multi-process dataloaders)
#bank.share_memory_()
#
## 4) Migrate to GPU and query
#bank.to("cuda:0")
#q = torch.randn(3, vocab.dim, device="cuda")  # 3 queries
#idx, scores, tokens = bank.nearest(q, k=8, device="cuda:0", return_tokens=True)
#print(tokens[0])  # top-8 tokens for query 0
#
## 5) Sharded across GPUs
#rows, vals, toks = bank.nearest_sharded(q, devices=["cuda:0", "cuda:1"], k=16)
#