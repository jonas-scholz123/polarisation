#%%
from dataloader import Dataloader

if __name__ == "__main__":
    network, id_to_sub = Dataloader().get_network()
    sub_to_id = {v: k for k,v in id_to_sub.items()}

def get_strongest_overlaps(sub):
    subsec = network[sub_to_id[sub]]
    tuples =  sorted(enumerate(subsec), key = lambda x: x[1], reverse = True)
    return [(id_to_sub[i], overlap) for i, overlap in tuples]

# %%
get_strongest_overlaps("politics")
#%%

pairs = []

for i, row in enumerate(network):
    for j, val in enumerate(row):
        if val != 0:
            pairs.append((i, j, val))

# %%
