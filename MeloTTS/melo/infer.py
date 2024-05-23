import os
import click
from melo.api import TTS



@click.command()
@click.option('--ckpt_path', '-m', type=str, default=None, help="Path to the checkpoint file")
@click.option('--text', '-t', type=str, default=None, help="Text to speak")
@click.option('--language', '-l', type=str, default="EN", help="Language of the model")
@click.option('--output_dir', '-o', type=str, default="outputs", help="Path to the output")
@click.option('--config_path', '-c', type=str, default=None, help="Path to the config")
def main(ckpt_path, text, language, output_dir, config_path):
    if ckpt_path is None:
        raise ValueError("The model_path must be specified")

    print(config_path)
    if config_path is None:
        config_path = os.path.join(os.path.dirname(ckpt_path), 'config.json')

    model = TTS(language=language, ckpt_path=ckpt_path)

    for spk_name, spk_id in model.hps.data.spk2id.items():
        save_path = f'{output_dir}/{spk_name}/output.wav'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.tts_to_file(text, spk_id, save_path)

if __name__ == "__main__":
    main()
