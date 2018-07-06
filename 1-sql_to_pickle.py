import pickle
import sqlite3

# @todo : add training db from taop-data
conn = sqlite3.connect('../db/training.db')
c = conn.cursor()

# @todo : try with og/fg/ibu/srm

max_fermentables = 5
max_hops = 9
training_data_percentage = 100


def get_recipe():
    # placeholder for styleID label in first row/col
    matrix = [[0, 0, 0]]
    for _ in range(max_fermentables):
        matrix.append([0, 0, 0])
    for _ in range(max_hops):
        matrix.append([0, 0, 0])
    return matrix


# get recipes
recipes = []
labels = []
yeast_data = []
hops_data = []
fermentables_data = []
c.execute("select recipeID, styleID from training_set order by random()")
results = c.fetchall()
print(str(len(results)) + " recipes fetched")
for (recipeID, styleID) in results:
    # get blank recipe
    recipe = get_recipe()

    # yeast
    row = 0
    c.execute("select yeastID from recipe_yeast where recipeID = ?", (recipeID,))
    yeast = c.fetchone()[0]
    recipe[row][0] = yeast

    # fermentables
    row = 1
    c.execute("select fermentableID, pounds from recipe_fermentables where recipeID = ? order by fermentableID",
              (recipeID,))
    fermentables = c.fetchall()

    fermentable_data = []
    for _ in range(max_fermentables):
        fermentable_data.append([0, 0])
    if len(fermentables) > max_fermentables:
        continue
    fermentable_row = 0
    for (fermentableID, fermentableAmount) in fermentables:
        recipe[row][0] = fermentableID
        recipe[row][1] = fermentableAmount
        fermentable_data[fermentable_row][0] = fermentableID
        fermentable_data[fermentable_row][1] = fermentableAmount
        row += 1
        fermentable_row += 1

    # hops
    row = max_fermentables + 1
    c.execute("select hopID, ounces, minutes from recipe_hops where recipeID = ? order by hopID", (recipeID,))
    hops = c.fetchall()

    hop_data = []
    for _ in range(max_hops):
        hop_data.append([0, 0, 0])
    if len(hops) > max_hops:
        continue
    hop_row = 0
    for (hopID, hopAmount, hopTime) in hops:
        recipe[row][0] = hopID
        recipe[row][1] = hopAmount
        recipe[row][2] = hopTime
        hop_data[hop_row][0] = hopID
        hop_data[hop_row][1] = hopAmount
        hop_data[hop_row][2] = hopTime
        row += 1
        hop_row += 1

    recipes.append(recipe)
    labels.append(styleID)
    yeast_data.append(yeast)
    hops_data.append(hop_data)
    fermentables_data.append(fermentable_data)

print("picking ... ", end="")
# divide into training and testing sets
total = len(recipes)
split = int((total * training_data_percentage) / 100)
train_data = recipes[0:split]
train_labels = labels[0:split]
# test_data = recipes[split+1:total-1]
# test_labels = labels[split+1:total-1]
pickle.dump(labels, open("data/labels.p", "wb"))
pickle.dump(train_data, open("data/train_data.p", "wb"))
# pickle.dump(test_data, open("data/test_data.p", "wb"))
pickle.dump(train_labels, open("data/train_labels.p", "wb"))
# pickle.dump(test_labels, open("data/test_labels.p", "wb"))

pickle.dump(yeast_data, open("data/yeast.p", "wb"))
pickle.dump(hops_data, open("data/hops.p", "wb"))
pickle.dump(fermentables_data, open("data/fermentables.p", "wb"))

conn.close()
print("ricked!")
