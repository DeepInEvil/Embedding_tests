from collections import defaultdict

with open('perf.txt', 'r') as f:
    perfs = f.readlines()

domains = ['Video_Games', 'Books', 'Toys_Games', 'Tools_Home_Improvement', 'Amazon_Instant_Video', 'Movies_TV', 'Electronics', 'Health',
           'Shoes', 'Baby', 'Automotive', 'Software', 'Sports_Outdoors', 'Clothing_Accessories', 'Beauty', 'Patio', 'Music',
           'Pet_Supplies', 'Office_Products', 'Home_Kitchen']
#word_embeddings = ['embeddings_snap_s256_e15.txt', 'embeddings_snap_s256_e50.txt', 'embeddings_snap_s256_e30.txt',
                     # 'embeddings_snap_s512_e15.txt', 'embeddings_snap_s128_e15.txt', 'embeddings_snap_s128_e30.txt',
                     # 'embeddings_snap_s128_e50.txt', 'embeddings_snap_s512_e50.txt', 'embeddings_snap_s512_e30.txt']
word_embeddings = ['embeddings_snap_s512_e15.txt', 'embeddings_snap_s128_e15.txt', 'embeddings_snap_s128_e30.txt',
                   'embeddings_snap_s128_e50.txt', 'embeddings_snap_s512_e50.txt', 'embeddings_snap_s512_e30.txt']

it = 0
perf_dict = defaultdict(list)
for i, emb in enumerate(word_embeddings):
    for j, domain in enumerate(domains):
        perf = perfs[it].replace('\n', '').strip().split(':')[-1][0:5]
        it = it + 1
        perf_dict[domain].append(perf)

for dom in domains:
    print (dom + ':' +'&'.join(perf_dict[dom]))


