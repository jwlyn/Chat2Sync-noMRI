from aip import AipSpeech
# import ffmpy3
import pygame
import time
from pydub import AudioSegment
import os


def word_to_audio(word):
    APP_ID = '23566670'
    API_KEY = 'FRMvC7896XeR47h9GTHFqjmz'
    SECRET_KEY = 'cfXnLLlAaUeMfpD6fcMKfVZqGGToHnKp'

    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

    result = client.synthesis(word, 'zh', 1, {'vol': 5, 'per': 3})

    # 识别正确返回语音二进制 错误则返回dict 参照下面错误码
    if not isinstance(result, dict):
        with open('audio/audio.mp3', 'wb') as f:
            f.write(result)

    # ff = ffmpy3.FFmpeg(executable='D:\\001_program\\ffmpeg\\bin\\ffmpeg.exe',
    #                    inputs={'audio/audio.mp3': None},
    #                    outputs={'audio/audio.wav': None},
    #                    global_options='-y')
    # ff.run()
    MP3_File = AudioSegment.from_mp3(file='audio/audio.mp3')
    MP3_File.export('audio/audio.wav', format="wav")


if __name__ == "__main__":
    pygame.mixer.init()
    while 1:
        word = input("input:\n      ")
        word_to_audio(word)
        pygame.mixer.init()
        pygame.mixer.music.load('audio/audio.wav')  # 加载音乐
        pygame.mixer.music.play()  # 播放
        pygame.quit()
