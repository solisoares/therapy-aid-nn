from pathlib import Path
# import wandb


def download_weights(model_url: str, out_path: Path):
    """ Download best model's weights from wandb

    Args:
        model_url (str): wandb model url
        out_path (Path): Location to download weights 
    """
    run = wandb.init()
    artifact = run.use_artifact(model_url, type='model')
    artifact.download(out_path)


if __name__ == "__main__":
    try:
        model_url = ... # add model url
        out_path = 'nn/3objs/weights'
        download_weights(model_url, out_path)
    except:
        print("--- Couldn't download weights. Check if WandB URL model is correct ---")
