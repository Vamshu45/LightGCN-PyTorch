import sys
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'lgn'
sys.argv = ['gen_recs.py', '--dataset=github', f'--model={MODEL}', '--layer=3',
            '--recdim=64', '--seed=2020', '--testbatch=18', '--tensorboard=0']
import json, numpy as np, torch
import world, utils, register
from register import dataset

K = 20
model = register.MODELS[world.model_name](world.config, dataset).to(world.device)
model.load_state_dict(torch.load(utils.getFileName(), map_location=world.device))
model.eval()

n_users, n_items = dataset.n_users, dataset.m_items
train = dataset.getUserPosItems(list(range(n_users)))
test = dataset.testDict
with torch.no_grad():
    scores = model.getUsersRating(torch.arange(n_users).long().to(world.device)).cpu().numpy()

pop = np.zeros(n_items)
for u in range(n_users): pop[train[u]] += 1
rng = np.random.RandomState(2020)
score_sets = {MODEL: scores,
              'popularity': np.tile(pop, (n_users, 1)),
              'random': rng.rand(n_users, n_items)}

def evaluate(sc):
    sc = sc.copy(); recs = {}; R = P = N = 0.0
    for u in range(n_users):
        sc[u, train[u]] = -np.inf
        top = np.argsort(-sc[u])[:K]; recs[u] = [int(i) for i in top]
        gt = set(test[u]); hit = np.array([i in gt for i in top], float)
        disc = 1 / np.log2(np.arange(2, K + 2))
        R += hit.sum() / len(gt); P += hit.sum() / K
        N += (hit * disc).sum() / disc[:min(len(gt), K)].sum()
    return R / n_users, P / n_users, N / n_users, recs

for name, sc in score_sets.items():
    r, p, n, recs = evaluate(sc)
    print(f"{name:11s} recall@20={r:.4f} precision@20={p:.4f} ndcg@20={n:.4f}")
    json.dump(recs, open(f"../recs_{name}.json", "w"))