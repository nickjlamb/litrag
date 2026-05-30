# Sample corpus

Static sample of PubMed abstracts so the repo runs key-free for ingest + retrieval.

TODO (build session): populate with ~10–20 real PubMed abstracts on one focused topic
(e.g. semaglutide / GLP-1 outcomes), each as:

## PMID: 00000000
**Title:** ...
**Source:** PubMed
...abstract text...

Pull them via PubCrawl (`search_pubmed` + `get_abstract`) and paste here, or use the
`--from-pubcrawl` ingest path once implemented. Pick a topic where some questions are
answerable and some deliberately are not — the unanswerable ones are what prove the
faithfulness eval works.
