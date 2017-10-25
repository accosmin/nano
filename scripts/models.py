def make_layer(name, params, activation):
        return "{}:{};{}{}".format(name, params, activation, ";" if activation else "")

def make_affine_layer(omaps, orows = 1, ocols = 1, activation = "act-snorm"):
        return make_layer("affine", "omaps={},orows={},ocols={}".format(omaps, orows, ocols), activation)

def make_output_layer(omaps, orows = 1, ocols = 1):
        return make_affine_layer(omaps, orows, ocols, "")

def make_norm_by_plane_layer():
        return make_layer("norm", "type={}".format("plane"))

def make_norm_globally_layer():
        return make_layer("norm", "type={}".format("global"))

def make_conv3d_layer(omaps, krows, kcols, kconn = 1, kdrow = 1, kdcol = 1, activation = "act-snorm"):
        return make_layer("conv3d", "omaps={},krows={},kcols={},kconn={},kdrow={},kdcol={}".format(omaps, krows, kcols, kconn, kdrow, kdcol), activation)
