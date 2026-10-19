# Proyek Sistem Rekomendasi Tiro
git clone [https://github.com/USERNAME/REPOSITORY_NAME.git](https://github.com/USERNAME/REPOSITORY_NAME.git)
cd REPOSITORY_NAME
python -m venv venv
# Di Linux/macOS:
source venv/bin/activate
# Di Windows:
venv\Scripts\activate

Install the Python packages used by the W6 runner:

```powershell
python -m pip install numpy pandas scipy
```
## W6: Bias Popularitas & Long Tail

Run the baseline and reranking sanity check from the repository root:

```powershell
python run_baselines.py
```

The pipeline derives popularity, rating-matrix factorization, TF-IDF vocabulary/IDF,
and user content profiles from training data. Validation is used only for
metrics and recommendation examples. It ranks the warm candidate catalog after
removing items already interacted with by each user. Relevance is rating >= 4,
the reported cutoff is Top-10, and long-tail means at most five training
interactions. Results are written to `outputs/w6/` (`config.json`,
`metrics.json`, `per_user_ndcg.csv`, and `top_n_examples.csv`).

The initial reranker uses a z-normalized MF score, subtracts normalized log
popularity, and adds a fixed bonus to items with at most five training
interactions. Its penalty and bonus are recorded in the experiment config.
MF uses truncated SVD over the sparse explicit 1–5 rating matrix. TF-IDF
uses `title` and `content_text_basic`; item popularity is the training
interaction count.
