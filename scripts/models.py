def make_layer(name, params, activation):
        return "{}:{};{}{}".format(name, params, activation, ";" if activation else "")

def make_affine_layer(omaps, orows = 1, ocols = 1, activation = "act-snorm"):
        name = "affine"
        params = "omaps={},orows={},ocols={}".format(omaps, orows, ocols)
        return make_layer(name, params, activation)

def make_output_layer(omaps, orows = 1, ocols = 1):
        return make_affine_layer(omaps, orows, ocols, "")

def make_norm_by_plane_layer():
        name = "norm"
        params = "type={}".format("plane")
        return make_layer(name, params)

def make_norm_globally_layer():
        name = "norm"
        params = "type={}".format("global")
        return make_layer(name, params)

def make_conv3d_layer(omaps, krows, kcols, kconn = 1, kdrow = 1, kdcol = 1, activation = "act-snorm"):
        name = "conv3d"
        params = "omaps={},krows={},kcols={},kconn={},kdrow={},kdcol={}".format(omaps, krows, kcols, kconn, kdrow, kdcol)
        return make_layer(name, params, activation)
