from research.comparisons.models import build


def build_model(input_dim, num_classes, params):
    return build('crosshar', input_dim, num_classes, params)
