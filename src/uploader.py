from omegaconf import OmegaConf, open_dict
from huggingface_hub import HfApi, Repository
import nemo.collections.asr as nemo_asr
from loguru import logger
from classopt import classopt, config
import os
from dataclasses import dataclass
from typing import List, Optional, Any
from readme_template import get_template

model_classes = {
    "EncDecCTCModel": nemo_asr.models.EncDecCTCModel,
    "EncDecRNNTModel": nemo_asr.models.EncDecRNNTModel,
}


@classopt
class Args:
    model_path: str = config(help="NeMo Model Path")
    user_name: str = config(default="TUT-SLP-lab", help="HuggingFace Username")
    model_type: str = config(
        default="EncDecCTCModel", choices=list(model_classes.keys()), help="Model Type"
    )
    tags: str = config(nargs="*", type=str, help="Tags for model")
    datasets: str = config(nargs="*", type=str, help="Datasets for model")
    language: str = config(default="ja", help="model language")
    token: str = config(help="HuggingFace Token")
    commit_message: str = config(default="Initial commit!", help="Commit Message")
    create_new_repo: bool = config(
        default=False, help="Create a new repo or not", action="store_true"
    )


@dataclass
class NeMoHuggingFaceModelConfig:
    language: List[str]
    license: str
    datasets: List[str]
    thumbnail: Optional[str]
    tags: List[str]
    model_index: Any
    library_name: str = "nemo"


def uploader(args: Args):
    logger.info("Load model")
    model = model_classes[args.model_type].restore_from(restore_path=args.model_path)
    MODEL_NAME = os.path.basename(args.model_path).split(".")[0]
    hf_model_name = f"{args.user_name}/{MODEL_NAME}"

    # HF API
    api = HfApi()
    if args.create_new_repo:
        try:
            api.create_repo(
                repo_id=MODEL_NAME, repo_type="model", private=True, token=args.token
            )
            logger.info("Successfully created repository !")
        except Exception as e:
            logger.error(
                f"Repository is possibly already created. Refer to error here - \n\n{e}"
            )

    config = NeMoHuggingFaceModelConfig(
        language=[args.language],
        license="cc-by-4.0",
        datasets=args.datasets,
        thumbnail=None,
        tags=[
            "automatic-speech-recognition",
            "speech",
            "audio",
            "NeMo",
            "pytorch",
        ]
        + args.tags,
        model_index=[dict(name=MODEL_NAME, results=[])],
    )
    config = OmegaConf.structured(config)

    with open_dict(config):
        config["model-index"] = config.pop("model_index")
        normalized_datasets = [ds_name.replace(" ", "-") for ds_name in args.datasets]
        config["datasets"] = OmegaConf.create(normalized_datasets)

    with Repository(
        local_dir=os.path.dirname(args.model_path),
        clone_from=hf_model_name,
        repo_type="model",
    ).commit("Upload README.md"):
        model.save_to(os.path.dirname(args.model_path))
        with open("readme_template.md", "w") as f:
            f.write("---\n")
            f.write(OmegaConf.to_yaml(config))
            f.write("\n---\n\n")
            f.write(get_template(hf_model_name))


if __name__ == "__main__":
    args: Args = Args.from_args()
    uploader(args)
