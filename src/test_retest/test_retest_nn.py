from saved_model.binary_classifier import load_pretrained_classifier
from saved_model.prepare_resnet50 import prepare_resnet_model
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.test_retest.test_retest_utils import get_cat, get_contingency_table, load_image_paths, preprocess


from utils import get_torch_device


image_dir = Path("./data/before_after_images/processed")
before_dir = image_dir / 'before'
after_dir = image_dir / 'after'
results_path = Path("./results/batch_nn_preds.csv")


def full_nn_pipeline(x, resnet_model=None, binary_classifier=None):

    if resnet_model == None:
        resnet_model = prepare_resnet_model(
            "./saved_model/resnet50_ft_weight.pkl")

    if binary_classifier == None:
        binary_classifier = load_pretrained_classifier(
            './saved_model/weights-2.pth')

    device = get_torch_device()
    x = torch.Tensor(x.transpose(0, 3, 1, 2))  # 1x3x224x224
    x = x.to(device)
    x = resnet_model(x)
    x = torch.sigmoid(binary_classifier(x))
    x = torch.round(x)
    return x.detach().cpu().numpy()


if __name__ == "__main__":

    device = get_torch_device()

    results = pd.DataFrame({'sample_paths': before_dir})

    results['nn_results_1'] = np.nan
    results['nn_results_2'] = np.nan

    before_paths = load_image_paths(before_dir)
    after_paths = load_image_paths(after_dir)

    for i, (before_path, after_path) in tqdm(enumerate(zip(before_paths, after_paths)), total=len(before_paths)):
        if before_path.stem != after_path.stem:
            print(
                f"Before and after don't match for index {i}, before: {before_path}, after: {after_path}")
            break

        results.loc[i, 'results_1'] = full_nn_pipeline(
            preprocess(before_path)).squeeze()
        results.loc[i, 'results_2'] = full_nn_pipeline(
            preprocess(after_path)).squeeze()

    total = len(results)
    results.to_csv(results_path)

    # pred_results = pd.read_csv(results_path, index_col=0)

    results_1 = results['results_1']
    results_2 = results['results_2']
    fns = results['sample_paths']

    # extract filename without extension from path
    fns = fns.map(lambda fn: Path(fn).stem)

    categories = fns.map(lambda fn: fn.split()[0])

    categories = categories.map(get_cat)

    idx = np.where(categories == 'images')[0]

    categories[idx] = fns[idx].map(lambda s: s.split()[2])

    cat_names = categories.unique()

    nn_contingency = get_contingency_table(
        pred_results=results, before_colname="results_1", after_colname="results_2", category_names=cat_names, categories=categories)

    nn_contingency.to_csv('./results/nn_contingency.csv')
