
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(), logging.FileHandler("train.log")],
)
log = logging.getLogger(__name__)


class Config:
    en_model_path = "opensubtitles_en_es/word2vec_en.model"
    es_model_path = "opensubtitles_en_es/word2vec_es.model"
    out_dir = "aligned"

    # vocabulary restriction (gensim orders vocab by frequency, most frequent first)
    vocab_size = 50_000          # how many of the most frequent words per language to use
    dis_most_frequent = 50_000   # restrict discriminator sampling to this many top words

    # adversarial training
    emb_dim = None                # inferred from the models
    n_epochs = 5
    epoch_size = 500_000           # number of word pairs "seen" per epoch
    batch_size = 32
    lr = 0.1                       # plain SGD, as in the original MUSE paper, the learning rate is quite high here
    # learning rate high because the matrices are quite small, (W is just 300 x 300), meaning it is more tolerant to divergence. 
    map_beta = 0.01                # orthogonalization strength applied after each mapping step
    log_every = 2000               # log running losses every this many generator steps

    # discriminator
    dis_hid_dim = 2048          # dimensions for matrices
    dis_layers = 2
    dis_dropout = 0.1  # changes 10% of the neurons to 0, preventing it from giving absolute yes/no answers and having more nuance in the gradient
    dis_input_dropout = 0.1
    dis_smooth = 0.1               # turns the targets from 1s/0s into 0.9s/0.1s.  As this model uses a sigmoid activation function, to achieve a certaintiy of 1 the weight would need to be infinitely high.  Having a limit at 0.9 gives a limit to the weights.  Plus, the more certain our model is, the less information it can give to W 
    dis_lambda = 1.0  
    dis_steps = 5                  # discriminator steps per generator step

    # CSLS / dictionary building
    csls_knn = 10                     # number of nearest neighbours that are considered to be part of the 'hub'
    csls_chunk = 2048              # rows processed at a time when computing CSLS (memory control)
    dico_max_rank = 15_000         # only the top-N most frequent words are searched when building the dictionary
    dico_max_size = 20_000         # max number of pairs kept in the induced dictionary
    dico_min_cos = 0.0             # discard pairs below this CSLS score

    # Procrustes refinement
    n_refinement = 5

    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"   


cfg = Config()

if os.environ.get("SMOKE_TEST"):
    cfg.vocab_size = 5_000
    cfg.dis_most_frequent = 5_000
    cfg.dico_max_rank = 5_000
    cfg.epoch_size = 50_000
    cfg.n_epochs = 1
    cfg.n_refinement = 1
    cfg.log_every = 200
    cfg.out_dir = "aligned_smoke"      

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)


def load_embeddings(model_path, vocab_size):
    log.info(f"Loading {model_path} ...")
    model = Word2Vec.load(model_path)
    wv = model.wv
    words = wv.index_to_key[:vocab_size]
    vectors = np.stack([wv[w] for w in words]).astype(np.float32)
    log.info(f"  -> kept {len(words):,} most frequent words, dim={vectors.shape[1]}")
    return words, vectors


def normalize_embeddings(emb):
    for _ in range(2):
        emb = emb - emb.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1
        emb = emb / norms
    return emb.astype(np.float32)


class Mapping(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.W.weight.data.copy_(torch.diag(torch.ones(dim)))  # init as identity

    def forward(self, x):
        return self.W(x)

    def orthogonalize(self, beta):
        """W <- (1 + beta) W - beta * (W W^T) W   (keeps W close to orthogonal)."""
        W = self.W.weight.data
        self.W.weight.data = (1 + beta) * W - beta * W.mm(W.t().mm(W))


class Discriminator(nn.Module):
    def __init__(self, dim, hid_dim, n_layers, dropout, input_dropout):
        super().__init__()
        layers = [nn.Dropout(input_dropout)]
        in_dim = dim
        for i in range(n_layers):
            out_dim = hid_dim if i < n_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)


def train_adversarial(src_emb, tgt_emb, cfg):
    device = cfg.device
    dim = src_emb.shape[1]
    src_t = torch.from_numpy(src_emb).to(device)
    tgt_t = torch.from_numpy(tgt_emb).to(device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    mapping = Mapping(dim).to(device)
    discriminator = Discriminator(
        dim, cfg.dis_hid_dim, cfg.dis_layers, cfg.dis_dropout, cfg.dis_input_dropout
    ).to(device)

    opt_map = torch.optim.SGD(mapping.parameters(), lr=cfg.lr)
    opt_dis = torch.optim.SGD(discriminator.parameters(), lr=cfg.lr)

    n_src = min(cfg.dis_most_frequent, src_t.size(0))
    n_tgt = min(cfg.dis_most_frequent, tgt_t.size(0))

    smooth = cfg.dis_smooth

    for epoch in range(cfg.n_epochs):
        n_steps = cfg.epoch_size // cfg.batch_size
        dis_losses, map_losses = [], []
        for step in range(n_steps):
            for _ in range(cfg.dis_steps):
                src_ids = torch.randint(0, n_src, (cfg.batch_size,), device=device)
                tgt_ids = torch.randint(0, n_tgt, (cfg.batch_size,), device=device)
                with torch.no_grad():
                    src_batch = mapping(src_t[src_ids])
                tgt_batch = tgt_t[tgt_ids]

                x = torch.cat([src_batch, tgt_batch], 0)
                # labels: 1 = real target, 0 = mapped source (with label smoothing)
                y = torch.cat([
                    torch.full((cfg.batch_size,), smooth, device=device),
                    torch.full((cfg.batch_size,), 1 - smooth, device=device),
                ])
                preds = discriminator(x)
                loss_d = nn.functional.binary_cross_entropy(preds, y)

                opt_dis.zero_grad()
                loss_d.backward()
                opt_dis.step()
                dis_losses.append(loss_d.item())

            src_ids = torch.randint(0, n_src, (cfg.batch_size,), device=device)
            tgt_ids = torch.randint(0, n_tgt, (cfg.batch_size,), device=device)
            src_batch = mapping(src_t[src_ids])
            tgt_batch = tgt_t[tgt_ids]
            x = torch.cat([src_batch, tgt_batch], 0)
            # flip labels: train mapping to fool D into thinking mapped==real
            y = torch.cat([
                torch.full((cfg.batch_size,), 1 - smooth, device=device),
                torch.full((cfg.batch_size,), smooth, device=device),
            ])
            preds = discriminator(x)
            loss_m = cfg.dis_lambda * nn.functional.binary_cross_entropy(preds, y)

            opt_map.zero_grad()
            loss_m.backward()
            opt_map.step()
            mapping.orthogonalize(cfg.map_beta)
            map_losses.append(loss_m.item())

            if (step + 1) % cfg.log_every == 0:
                log.info(
                    f"[adversarial] epoch {epoch+1}/{cfg.n_epochs} step {step+1}/{n_steps} "
                    f"dis_loss={np.mean(dis_losses[-cfg.log_every * cfg.dis_steps:]):.4f} "
                    f"map_loss={np.mean(map_losses[-cfg.log_every:]):.4f}"
                )

        log.info(
            f"[adversarial] epoch {epoch+1}/{cfg.n_epochs} DONE "
            f"dis_loss={np.mean(dis_losses):.4f} map_loss={np.mean(map_losses):.4f}"
        )
        # checkpoint after every epoch so a crash never loses everything
        np.save(
            os.path.join(cfg.out_dir, "W_adv_last_epoch.npy"),
            mapping.W.weight.data.cpu().numpy(),
        )

    W = mapping.W.weight.data.cpu().numpy()
    return W


def knn_avg_sim(queries, keys, k, chunk=2048):
    """
    For each row of `queries`, the mean cosine similarity to its k nearest rows
    of `keys` (both unit-normalized). Done in chunks, so memory is
    chunk x len(keys) floats rather than len(queries) x len(keys).
    """
    q = torch.from_numpy(np.ascontiguousarray(queries))
    kt = torch.from_numpy(np.ascontiguousarray(keys)).t().contiguous()
    out = []
    for i in range(0, q.size(0), chunk):
        sims = q[i:i + chunk].mm(kt)
        out.append(sims.topk(k, dim=1)[0].mean(dim=1))
    return torch.cat(out).numpy()


def csls_scores(src_emb, tgt_emb, knn, chunk=2048):
    """
    Cross-Domain Similarity Local Scaling.
    Corrects plain cosine similarity for 'hub' points (vectors that are
    nearest neighbor to many others), which otherwise dominate nearest-
    neighbor lookups and hurt dictionary quality.

    Returns a (len(src), len(tgt)) matrix, so keep the inputs to ~20k rows each
    (20k x 20k float32 = 1.6 GB).
    """
    r_src = torch.from_numpy(knn_avg_sim(src_emb, tgt_emb, knn, chunk))   # per source word
    r_tgt = torch.from_numpy(knn_avg_sim(tgt_emb, src_emb, knn, chunk))   # per target word

    src = torch.from_numpy(np.ascontiguousarray(src_emb))
    tgt = torch.from_numpy(np.ascontiguousarray(tgt_emb))
    scores = src.mm(tgt.t())                                    # cosine sim (unit-normalized)
    scores.mul_(2).sub_(r_src.unsqueeze(1)).sub_(r_tgt.unsqueeze(0))   # in place: no extra copies
    return scores.numpy()


def build_dictionary(src_emb, tgt_emb, W, cfg):
    """Map src -> tgt space with W, then induce a dictionary via CSLS
    mutual-nearest-neighbor matching. Only the top `dico_max_rank` most
    frequent words of each language are searched (rows are frequency-sorted)."""
    n = min(cfg.dico_max_rank, src_emb.shape[0], tgt_emb.shape[0])
    mapped_src = (src_emb[:n] @ W.T).astype(np.float32)
    scores = csls_scores(mapped_src, tgt_emb[:n], cfg.csls_knn, cfg.csls_chunk)

    best_tgt_for_src = scores.argmax(axis=1)
    best_src_for_tgt = scores.argmax(axis=0)

    pairs = []
    for i, j in enumerate(best_tgt_for_src):
        if best_src_for_tgt[j] == i:          # mutual nearest neighbors only
            pairs.append((i, j, scores[i, j]))

    pairs.sort(key=lambda p: -p[2])
    pairs = [p for p in pairs if p[2] >= cfg.dico_min_cos][: cfg.dico_max_size]
    log.info(f"  induced dictionary size: {len(pairs):,} mutual-NN pairs (CSLS)")
    return np.array([[p[0], p[1]] for p in pairs])


def procrustes(src_emb, tgt_emb, dico):
    """Closed-form solution: W* = U V^T, where U,S,V = SVD(Y^T X)."""
    X = src_emb[dico[:, 0]]
    Y = tgt_emb[dico[:, 1]]
    M = Y.T.dot(X)
    U, _, Vt = np.linalg.svd(M)
    W = U.dot(Vt)
    return W.astype(np.float32)


def refine(src_emb, tgt_emb, W, cfg):
    for i in range(cfg.n_refinement):
        dico = build_dictionary(src_emb, tgt_emb, W, cfg)
        if len(dico) < 10:
            log.warning("Dictionary too small to refine further, stopping.")
            break
        W = procrustes(src_emb, tgt_emb, dico)
        log.info(f"[procrustes] refinement iter {i+1}/{cfg.n_refinement} done")
        np.save(os.path.join(cfg.out_dir, "W_refined_last_iter.npy"), W)
    return W

class Translator:
    def __init__(self, npz_path, knn=10):
        data = np.load(npz_path, allow_pickle=True)
        self.src_words = list(data["src_words"])
        self.tgt_words = list(data["tgt_words"])
        self.src_emb = data["src_emb"]
        self.tgt_emb = data["tgt_emb"]
        self.W = data["W"]
        self.knn = knn
        self.mapped_src = (self.src_emb @ self.W.T).astype(np.float32)
        self.src_word2id = {w: i for i, w in enumerate(self.src_words)}
        # CSLS target-side term: how "hubby" each target word is w.r.t. the WHOLE
        # mapped source vocabulary. Computed once (about a minute on CPU for 50k words).
        self.r_tgt = knn_avg_sim(self.tgt_emb, self.mapped_src, knn)

    def translate(self, word, topn=5):
        if word not in self.src_word2id:
            raise KeyError(f"'{word}' not in source vocabulary")
        idx = self.src_word2id[word]
        sims = self.tgt_emb @ self.mapped_src[idx]                 # cosine sim to every target word
        r_src = np.partition(sims, -self.knn)[-self.knn:].mean()   # mean sim of query to its k nearest
        scores = 2 * sims - r_src - self.r_tgt
        best = np.argsort(-scores)[:topn]
        return [(self.tgt_words[i], float(scores[i])) for i in best]

def main():
    os.makedirs(cfg.out_dir, exist_ok=True)

    en_words, en_emb = load_embeddings(cfg.en_model_path, cfg.vocab_size)
    es_words, es_emb = load_embeddings(cfg.es_model_path, cfg.vocab_size)
    cfg.emb_dim = en_emb.shape[1]
    assert en_emb.shape[1] == es_emb.shape[1], "EN/ES vector_size must match (train both with same vector_size!)"

    en_emb = normalize_embeddings(en_emb)
    es_emb = normalize_embeddings(es_emb)

    adv_path = os.path.join(cfg.out_dir, "W_adversarial.npy")
    if os.path.exists(adv_path):
        log.info(f"=== Stage 1 skipped: loading finished adversarial W from {adv_path} ===")
        W = np.load(adv_path)
    else:
        log.info("=== Stage 1: adversarial training (EN -> ES) ===")
        W = train_adversarial(en_emb, es_emb, cfg)
        np.save(adv_path, W)
        log.info(f"Saved adversarial W to {adv_path}")

    log.info("=== Stage 2: iterative orthogonal Procrustes refinement ===")
    W = refine(en_emb, es_emb, W, cfg)

    out_path = os.path.join(cfg.out_dir, "en_es_mapping.npz")
    np.savez(
        out_path,
        W=W,
        src_words=np.array(en_words, dtype=object),
        tgt_words=np.array(es_words, dtype=object),
        src_emb=en_emb,
        tgt_emb=es_emb,
    )
    log.info(f"Saved mapping + embeddings to {out_path}")

    log.info("=== Sample translations (EN -> ES) ===")
    translator = Translator(out_path)
    test_words = ["dog", "house", "love", "good", "water", "king", "friend"]
    for w in test_words:
        try:
            preds = translator.translate(w, topn=5)
            log.info(f"  {w:10s} -> {preds}")
        except KeyError:
            log.info(f"  {w:10s} -> (not in vocabulary, skipped)")


if __name__ == "__main__":
    main()