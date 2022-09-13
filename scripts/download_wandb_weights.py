from pathlib import Path
import wandb


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
    model_url = 'solisoares/3objs/run_w9r9wfq2_model:v0'
    out_path = 'nn/3objs/weights'
    download_weights(model_url, out_path)
