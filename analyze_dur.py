import numpy as np

data = np.load('mouse_dataset_fixed_N256.npz')
masks = data['padding_masks'].astype(bool)
seqs = data['seq_tensors']
dts = []
episode_durs = []

for i in range(len(seqs)):
    valid_seq = seqs[i][masks[i]]
    # dt_ms = seq[:, 2] * 100
    ep_dts = valid_seq[:, 2] * 100
    dts.extend(ep_dts)
    episode_durs.append(np.sum(ep_dts))

dts = np.array(dts)
episode_durs = np.array(episode_durs)

print('=== Step DTs (ms) ===')
print('Mean:', np.mean(dts))
print('Median:', np.median(dts))
print('Min:', np.min(dts))
print('Max:', np.max(dts))
print('95th percentile:', np.percentile(dts, 95))
print('99th percentile:', np.percentile(dts, 99))

print('\n=== Episode Durations (ms) ===')
print('Mean:', np.mean(episode_durs))
print('Median:', np.median(episode_durs))
print('95th percentile:', np.percentile(episode_durs, 95))
print('99th percentile:', np.percentile(episode_durs, 99))
