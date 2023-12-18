import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

import hydra
import nemo.collections.asr as nemo_asr
from huggingface_hub import HfApi, Repository
from loguru import logger
from omegaconf import OmegaConf, open_dict

from readme_template import get_template
import subprocess

model_classes = {
    "EncDecCTCModel": nemo_asr.models.EncDecCTCModel,
    "EncDecRNNTModel": nemo_asr.models.EncDecRNNTModel,
}


@dataclass
class Args:
    model_path: str = ""  # NeMo Model Path
    organization: str = "TUT-SLP-lab"  # HuggingFace Username
    user_name: str = ""
    email: str = ""
    model_type: str = "EncDecCTCModel"  # Model Type
    tags: List[str] = field(default_factory=list)  # Tags for model"
    datasets: List[str] = field(default_factory=list)  # Datasets for model
    language: str = "ja"  # Model language
    token: str = ""  # HuggingFace Token
    commit_message: str = "Initial commit!"  # Commit Message
    create_new_repo: bool = True  # Store model in new repo or not


@dataclass
class NeMoHuggingFaceModelConfig:
    language: List[str]
    license: str
    datasets: List[str]
    thumbnail: Optional[str]
    tags: List[str]
    model_index: Any
    library_name: str = "nemo"


@hydra.main(version_base=None, config_path="../conf/", config_name="config")
def main(args: Args):
    logger.info("Load model")
    model = model_classes[args.model_type].restore_from(restore_path=args.model_path)
    MODEL_NAME = os.path.basename(args.model_path).split(".")[0]
    hf_model_name = f"{args.organization}/{MODEL_NAME}"
    logger.info(f"Repository name: {hf_model_name}")

    # HF api
    api = HfApi(endpoint="https://huggingface.co", token=args.token)
    if args.create_new_repo:
        try:
            logger.info(f"Creating {hf_model_name} repository...")
            api.create_repo(repo_id=hf_model_name, repo_type="model", private=True)
            logger.info("Successfully created repository!")
        except Exception as e:
            logger.error(f"Failed to create repository. Refer to error here - \n\n{e}")
            return

    logger.info("Push model to repository")
    tags = [
        "automatic-speech-recognition",
        "speech",
        "audio",
        "NeMo",
        "pytorch",
    ] + args.tags
    config = NeMoHuggingFaceModelConfig(
        language=[args.language],
        license="cc-by-4.0",
        datasets=args.datasets,
        thumbnail=None,
        tags=tags,
        model_index=[dict(name=MODEL_NAME, results=[])],
    )
    config = OmegaConf.structured(config)
    with open_dict(config):
        config["model-index"] = config.pop("model_index")
        normalized_datasets = [ds_name.replace(" ", "-") for ds_name in args.datasets]
        config["datasets"] = OmegaConf.create(normalized_datasets)

    with Repository(
        local_dir="/app/hf-model",
        clone_from=hf_model_name,
        repo_type="model",
        use_auth_token=args.token,
        git_user=args.user_name,
        git_email=args.email,
    ).commit(args.commit_message):
        model.save_to(os.path.basename(args.model_path))
        with open("README.md", "w") as f:
            f.write("---\n")
            f.write(OmegaConf.to_yaml(config))
            f.write("\n---\n\n")
            f.write(get_template(hf_model_name))
    logger.info(f"Successfully pushed model to {hf_model_name}!")


if __name__ == "__main__":
    main()
