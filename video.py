import numpy as np
import cv2
import pygame
import time
from inference import load_model, datagen
import audio
import torch
from aip import AipSpeech
from pydub import AudioSegment
import ffmpy3


device = 'cuda' if torch.cuda.is_available() else 'cpu'
mel_step_size = 16
fps = 25
print('Using {} for inference.'.format(device))


def net():
    APP_ID = '23566670'
    API_KEY = 'FRMvC7896XeR47h9GTHFqjmz'
    SECRET_KEY = 'cfXnLLlAaUeMfpD6fcMKfVZqGGToHnKp'

    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
    return client


def word2audio(client, word):
    result = client.synthesis(word, 'zh', 1, {'vol': 5, 'per': 3})

    # 识别正确返回语音二进制 错误则返回dict 参照下面错误码
    if not isinstance(result, dict):
        with open('audio/audio.mp3', 'wb') as f:
            f.write(result)

    ff = ffmpy3.FFmpeg(executable='C:\\ffmpeg\\bin\\ffmpeg.exe',
                       inputs={'audio/audio.mp3': None},
                       outputs={'audio/audio.wav': None},
                       global_options='-y')
    ff.run()

    # MP3_File = AudioSegment.from_mp3(file='audio/audio.mp3')
    # MP3_File.export('audio/audio.wav', format="wav")


def gen_data(audio_path, full_frames):
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]

    gen = datagen(full_frames.copy(), mel_chunks)
    return gen


def gen_fake(model, audio_path, full_frames):
    frameV = []
    for img_batch, mel_batch, frames, coords in gen_data(audio_path, [full_frames]):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            frameV.append(f)
    return frameV


def main(image_path, checkpoint_path):
    # 加载图像和模型
    image = cv2.imread(image_path)
    model = load_model(checkpoint_path)
    client = net()

    flag = 0
    while True:
        if flag:
            word = input("input:\n      ")
            if not len(word):
                continue
            word2audio(client, word)
            frames = gen_fake(model, 'audio/audio.wav', image)
            # 播放语音和视频
            pygame.mixer.init()
            pygame.mixer.music.load('audio/audio.wav')  # 加载音乐
            pygame.mixer.music.play()  # 播放
            for im in frames:
                cv2.imshow('frame', im)
                cv2.waitKey(1000//fps)
            pygame.quit()

        while True:
            frame = image
            # 显示结果帧e
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == 27:  # ord('q') enter:13 esc:27
                flag = 1
                break

    # pygame.mixer.music.stop()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main('image/trump.jpg',
         'checkpoints/wav2lip.pth',)
