from tts_rater.models import ReferenceEncoder
import torch
from tts_rater.mel_processing import spectrogram_torch
import librosa
import os
from easydict import EasyDict as edict
from torch.nn.functional import cosine_similarity, mse_loss
import glob
import numpy as np
import whisper
from jiwer import wer

hps = edict(
    {
        "data": {
            "sampling_rate": 22050,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
        }
    }
)


def extract_se(ref_enc, ref_wav_list, se_save_path=None, device="cpu"):
    if isinstance(ref_wav_list, str):
        ref_wav_list = [ref_wav_list]

    gs = []

    for fname in ref_wav_list:
        audio_ref, sr = librosa.load(fname, sr=hps.data.sampling_rate)
        y = torch.FloatTensor(audio_ref)
        y = y.to(device)
        y = y.unsqueeze(0)
        y = spectrogram_torch(
            y,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        ).to(device)
        with torch.no_grad():
            g = ref_enc(y.transpose(1, 2)).unsqueeze(-1)
            gs.append(g.detach())
    gs = torch.stack(gs).mean(0)

    if se_save_path is not None:
        os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
        torch.save(gs.cpu(), se_save_path)

    return gs.squeeze(-1)


ref_enc = ReferenceEncoder(1024 // 2 + 1, 256)
checkpoint = torch.load("tts_rater/reference_encoder.pth", map_location="cpu")
ref_enc.load_state_dict(checkpoint["model"], strict=True)
vec_gt = extract_se(ref_enc, glob.glob("tts_rater/examples/speaker1_en_us/*.wav"))


def compute_tone_color_similarity(audio_paths, vec_gt):
    scores = []
    for wav_gen in audio_paths:
        vec_gen = extract_se(ref_enc, wav_gen)
        score = cosine_similarity(vec_gen, vec_gt).item()
        scores.append(score)
    return np.mean(scores)


model = whisper.load_model("medium")


def compute_wer(texts, audio_paths):
    wer_results = []
    assert len(texts) == len(audio_paths)
    for text, audio_path in zip(texts, audio_paths):
        result = model.transcribe(audio_path)
        # print(result)
        wer_results.append(wer(text.strip(), result["text"]))
    return np.mean(wer_results)


# suppose there are audios generated by different models (in generated folder)
# NOTE: the texts and audio_paths must be aligned
texts = open("tts_rater/en_example_text.txt").readlines()


def rate(ckpt_path):
    from melo.api import TTS

    model = TTS(language="EN", device="auto", ckpt_path=ckpt_path)
    speaker_ids = model.hps.data.spk2id
    spkr = speaker_ids["EN-US"]

    for i, text in enumerate(texts):
        save_path = f"tmp/sent_{i:03d}.wav"
        # remove the directory if it exists
        if os.path.exists(save_path):
            os.remove(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.tts_to_file(text, spkr, save_path, speed=1.0)

    audio_paths = sorted(glob.glob("tmp/*.wav"))

    tone_color_sim = compute_tone_color_similarity(audio_paths, vec_gt)
    word_error_rate = compute_wer(texts, audio_paths)

    return (1 - tone_color_sim, word_error_rate)


if __name__ == "__main__":
    print(rate("tts_rater/MeloTTS-English/checkpoint.pth"))
