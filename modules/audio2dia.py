#!/usr/bin/env python3
# -*- coding: utf-8
# ----------------------------------------------------------------------------
# Created By  : Thang Le, Linh Vo Van
# Organization: ACM Lab
# Created Date: 2024/03/25
# version ='1.0'
# ---------------------------------------------------------------------------

import whisperx
import gc


class Audio2Dia:
    """
    Converts audio files to speaker-diarized dialogue transcripts.

    This class utilizes the Whisper library (whisperx) to transcribe audio,
    align text with speech, and perform speaker diarization. 

    Args:
        name_model (str): Name of the Whisper model to use for transcription.
        batch_size (int): Batch size for processing audio during transcription.
        device (str): Device to run the model on (e.g., "cpu" or "cuda").
        compute_type (str, optional): Compute type for the model ("fp16" or "fp32"). 
            Defaults to "fp16".
    """
    def __init__(self, name_model, batch_size, device, compute_type) -> None:
        self.model = whisperx.load_model(
            name_model, device, compute_type=compute_type)
        self.batch_size = batch_size
        self.device = device

    def _delete_model(self):
        """
        (Optional) Explicitly releases resources held by the model.

        This method is currently empty (`pass`) but serves as a placeholder
        for potential future resource management (e.g., GPU memory cleanup).
        """
        pass

    def _save_model_local(self):
        pass

    def generate(self, audio_file, file_save_dia):
        """
        Processes an audio file and saves speaker-diarized dialogue transcripts.

        Args:
            audio_file (str): Path to the audio file for transcription.
            file_save_dia (str): Path to the output file where transcripts are saved.
        """
        audio = whisperx.load_audio(audio_file)
        result = self.model.transcribe(audio,
                                       batch_size=self.batch_size)

        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=self.device)
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token="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", device=self.device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)


        result = whisperx.assign_word_speakers(diarize_segments, result)
        text_result =""
        for i in range(len(result["segments"])):
            try:
                text_result += str(result["segments"][i]["start"])+"--->"+str(result["segments"][i]["end"])+ "\t" + result["segments"][i]["speaker"] +":" + result["segments"][i]["text"] +"\n"
            except:
                text_result += str(result["segments"][i]["start"])+"--->"+str(result["segments"][i]["end"])+ "\t" + "None" +":" + result["segments"][i]["text"] +"\n"
        with open (file_save_dia,"w") as f:
            f.write(text_result)
        gc.collect()
