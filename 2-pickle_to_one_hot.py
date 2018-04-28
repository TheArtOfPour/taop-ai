import keras
import numpy
import pickle
import sqlite3
conn = sqlite3.connect('../db/training.db')
c = conn.cursor()
# fix random seed for reproducibility
numpy.random.seed(7)

max_fermentables = 5
max_hops = 9

yeast = numpy.array(pickle.load(open("data/yeast.p", "rb")))
hops = numpy.array(pickle.load(open("data/hops.p", "rb")))
fermentables = numpy.array(pickle.load(open("data/fermentables.p", "rb")))

c.execute("select MAX(id) AS max_id from styles")
max_style = c.fetchone()[0] + 1
c.execute("select MAX(id) AS max_id from yeast")
max_yeast = c.fetchone()[0] + 1
c.execute("select MAX(id) AS max_id from hops")
max_hop = c.fetchone()[0] + 1
c.execute("select MAX(id) AS max_id from fermentables")
max_fermentable = c.fetchone()[0] + 1

print(str(len(yeast)) + ' entries')
print(str(max_yeast) + ' yeast categories ...', end=" ")
one_hot_yeast = keras.utils.to_categorical(yeast, num_classes=max_yeast)
print('pitched!')

one_hot_hops = []
hop_amounts = []
hop_times = []
print(str(max_hop) + ' hop categories ...', end=" ")
for hop in hops:
    one_hot_hop = []
    single_hop_amount = []
    single_hop_time = []
    for hop_id, hop_amount, hop_time in hop:
        # hop amount in ounces 0:36 -> -1:1
        hop_amount = ((hop_amount/36) * 2) - 1
        # hop time in minutes 0:90 -> -1:1
        hop_time = ((hop_time/90) * 2) - 1
        # hop id in max_hop categories
        hop_id_hot = keras.utils.to_categorical(hop_id, num_classes=max_hop)[0]
        hop_id_hot = numpy.delete(hop_id_hot, 0)
        single_hop_amount.append(hop_amount)
        single_hop_time.append(hop_time)
        one_hot_hop.append(hop_id_hot)
    one_hot_hops.append(one_hot_hop)
    hop_amounts.append(single_hop_amount)
    hop_times.append(single_hop_time)
print('steeped!')

one_hot_fermentables = []
fermentable_amounts = []
print(str(max_fermentable) + ' fermentables categories ...', end=" ")
for fermentable in fermentables:
    one_hot_fermentable = []
    single_fermentable_amount = []
    for fermentable_id, fermentable_amount in fermentable:
        # fermentable amount in lbs 0:20 -> -1:1
        fermentable_amount = ((fermentable_amount/20) * 2) - 1
        # fermentable id in max_fermentable categories
        fermentable_id_hot = keras.utils.to_categorical(fermentable_id, num_classes=max_fermentable)[0]
        fermentable_id_hot = numpy.delete(fermentable_id_hot, 0)
        single_fermentable_amount.append(fermentable_amount)
        one_hot_fermentable.append(fermentable_id_hot)
    one_hot_fermentables.append(one_hot_fermentable)
    fermentable_amounts.append(single_fermentable_amount)
print('grained!')

# one_hot_combined = []
# print('with your powers ...', end=" ")
# for x in range(len(yeast)):
#     temp = []
#     yeast_id_hot = keras.utils.to_categorical(yeast[x], num_classes=max_yeast)[0]
#     yeast_id_hot = numpy.delete(yeast_id_hot, 0)
#     temp.append(yeast_id_hot)
#     for fermentable_id, fermentable_amount in fermentables[x]:
#         fermentable_id_hot = keras.utils.to_categorical(fermentable_id, num_classes=max_fermentable)[0]
#         fermentable_id_hot = numpy.delete(fermentable_id_hot, 0)
#         temp.append(fermentable_id_hot)
#     for hop_id, hop_amount, hop_time in hops[x]:
#         hop_id_hot = keras.utils.to_categorical(hop_id, num_classes=max_hop)[0]
#         hop_id_hot = numpy.delete(hop_id_hot, 0)
#         temp.append(hop_id_hot)
#     one_hot_combined.append(temp)
# print('combined!')

print('pickling...', end=" ")
pickle.dump(one_hot_yeast, open("data/one_hot_yeast.p", "wb"))
pickle.dump(one_hot_hops, open("data/one_hot_hops.p", "wb"))
pickle.dump(hop_amounts, open("data/hop_amounts.p", "wb"))
pickle.dump(hop_times, open("data/hop_times.p", "wb"))
pickle.dump(one_hot_fermentables, open("data/one_hot_fermentables.p", "wb"))
pickle.dump(fermentable_amounts, open("data/fermentable_amounts.p", "wb"))
# pickle.dump(one_hot_combined, open("data/one_hot_combined.p", "wb"))
print('ricked!')
