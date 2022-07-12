import os
import argparse
import torch
import numpy as np

from utils import compute_metrics
from utils import to_var, load_config_from_json

from torch.utils.data import DataLoader
from modcloth import ModCloth
from model import SFNet
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from random import sample


def display_pca_scatterplot(ids, embeddings, category, limit=50):

    # three_dim = PCA(random_state=0, n_components=2).fit_transform(embeddings)
    three_dim = TSNE(random_state=0, n_components=2, learning_rate='auto', init='random').fit_transform(np.array(embeddings))

    pprinter = {thing: n for n, thing in enumerate(set(sorted(category.values())))}
    

    data = []
    count = 0
    legends = []
    colors = ["aqua", "black", "blue", "blueviolet", "brown", "cadetblue", "chartreuse", "chocolate", "coral", "cornflowerblue", "cornsilk", "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgray", "darkgrey", "darkgreen", "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange", "darkorchid", "darkred", "darksalmon", "darkseagreen", "darkslateblue", "darkslategray", "darkslategrey", "darkturquoise", "darkviolet", "deeppink", "deepskyblue", "dimgray", "dimgrey", "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia", "gainsboro", "ghostwhite", "gold", "goldenrod", "gray", "grey", "green", "greenyellow", "honeydew", "hotpink", "indianred", "indigo", "ivory", "khaki", "lavender", "lavenderblush", "lawngreen", "lemonchiffon", "lightblue", "lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgray", "lightgrey", "lightgreen", "lightpink", "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray", "lightslategrey", "lightsteelblue", "lightyellow", "lime", "limegreen", "linen", "magenta", "maroon", "mediumaquamarine", "mediumblue", "mediumorchid", "mediumpurple", "mediumseagreen", "mediumslateblue", "mediumspringgreen", "mediumturquoise", "mediumvioletred", "midnightblue", "mintcream", "mistyrose", "moccasin", "navajowhite", "navy", "oldlace", "olive", "olivedrab", "orange", "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise", "palevioletred", "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue", "purple", "red", "rosybrown", "royalblue", "rebeccapurple", "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell", "sienna", "silver", "skyblue", "slateblue", "slategray", "slategrey", "snow", "springgreen", "steelblue", "tan", "teal", "thistle", "tomato", "turquoise", "violet", "wheat", "white", "whitesmoke", "yellow", "yellowgreen"]


    for id in sample(ids, limit):
        if category[id] in legends:
            trace = go.Scatter(
                x = [three_dim[count, 0]],
                y = [three_dim[count, 1]],
                # text = id,
                name = pprinter[category[id]],
                legendgroup = pprinter[category[id]],
                showlegend = False,
                textposition = "top center",
                textfont_size = 20,
                mode = 'markers+text',
                marker = {
                    'size': 10,
                    'opacity': 0.8,
                    'color': colors[legends.index(category[id])]
                }
            )
        else:
            legends.append(category[id])
            trace = go.Scatter(
                x = [three_dim[count, 0]],
                y = [three_dim[count, 1]],
                # text = id,
                name = pprinter[category[id]],
                legendgroup = pprinter[category[id]],
                # name = category[id],
                # legendgroup = category[id],
                textposition = "top center",
                textfont_size = 20,
                mode = 'markers+text',
                marker = {
                    'size': 10,
                    'opacity': 0.8,
                    'color': colors[legends.index(category[id])]
                }
            )

        data.append(trace)
        count += 1
        if count == limit:
            break
    
# Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1500,
        height = 900,
        )


    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.write_image("images/TSNE_cup_size_40000_bra_size_random.png")
    # plot_figure.show()


def main(args):

    data_config = load_config_from_json(args.data_config_path)
    model_config = load_config_from_json(
        os.path.join(args.model, "config.jsonl")
    )

    # initialize model
    model = SFNet(model_config["sfnet"])

    checkpoint = os.path.join(args.model, args.checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    print("Model loaded from %s" % (args.model))

    print("Preparing test data ...")
    dataset = ModCloth(data_config, split="full")

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=model_config["trainer"]["batch_size"],
        shuffle=False,
    )
    # waist, hips, bra_size, height, shoe_size


    user_values = {thing.detach().numpy().tolist(): model.user_embedding(thing).detach().numpy().tolist() for batch in data_loader for thing in batch['user_id']}
    category = {thing.detach().numpy().tolist(): comp[2].detach().numpy().tolist() for batch in data_loader for thing, comp in zip(batch['user_id'], batch['user_numeric'])}

    # user_values = {thing.detach().numpy().tolist(): model.cup_size_embedding(thing).detach().numpy().tolist() for batch in data_loader for thing in batch['cup_size']}

    # user_values = {thing.detach().numpy().tolist(): model.item_embedding(thing).detach().numpy().tolist() for batch in data_loader for thing in batch['item_id']}

    # user_values = {thing.detach().numpy().tolist(): model.category_embedding(thing).detach().numpy().tolist() for batch in data_loader for thing in batch['category']}

    print(len(category), len(set(category.values())))
    display_pca_scatterplot(list(user_values.keys()), list(user_values.values()), category, min(40000,len(user_values)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config_path", type=str, default="configs/data.jsonnet")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="E20.pytorch")

    args = parser.parse_args()
    main(args)
