

import customtkinter as ctk
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np
import simpleaudio as sa
import threading

# Konfiguracja pydub dla FFmpeg
# AudioSegment.converter = r"C:\\Users\\Damian\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-7.1.1-full_build\\bin\\ffmpeg.exe"
# AudioSegment.ffprobe = r"C:\\Users\\Damian\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-7.1.1-full_build\\bin\\ffprobe.exe"

# Globalna zmienna do odtwarzania
play_obj = None

class FilterDesignerWindow(ctk.CTkToplevel):
    def __init__(self, parent, initial_params, on_save):
        super().__init__(parent)
        self.title("Ustawienia: Filter Designer")
        self.geometry("400x300")
        self.on_save = on_save
        self.params = initial_params.copy()

        # Typ filtra
        self.filter_type_label = ctk.CTkLabel(self, text="Typ filtra:")
        self.filter_type_label.pack(pady=(10, 0))

        self.filter_type = ctk.CTkOptionMenu(
            self,
            values=["lowpass", "highpass", "bandpass", "notch"],
            command=self.update_filter_type
        )
        self.filter_type.set(self.params.get("filter_type", "lowpass"))
        self.filter_type.pack()

        # Częstotliwość graniczna
        self.freq_label = ctk.CTkLabel(self, text="Częstotliwość graniczna: 500 Hz")
        self.freq_label.pack(pady=(20, 0))

        self.freq_slider = ctk.CTkSlider(self, from_=20, to=20000, number_of_steps=1000,
                                         command=self.update_freq)
        self.freq_slider.set(self.params.get("cutoff_freq", 500.0))
        self.freq_slider.pack()

        # Q-factor
        self.q_label = ctk.CTkLabel(self, text="Q-factor: 1.0")
        self.q_label.pack(pady=(20, 0))

        self.q_slider = ctk.CTkSlider(self, from_=0.1, to=10.0, number_of_steps=99,
                                      command=self.update_q)
        self.q_slider.set(self.params.get("q_factor", 1.0))
        self.q_slider.pack()

        # Zapisz i zamknij
        self.save_btn = ctk.CTkButton(self, text="Zapisz i zamknij", command=self.save_and_close)
        self.save_btn.pack(pady=30)

    def update_filter_type(self, val):
        self.params["filter_type"] = val

    def update_freq(self, val):
        val = round(val)
        self.freq_label.configure(text=f"Częstotliwość graniczna: {val} Hz")
        self.params["cutoff_freq"] = val

    def update_q(self, val):
        val = round(val, 1)
        self.q_label.configure(text=f"Q-factor: {val}")
        self.params["q_factor"] = val

    def save_and_close(self):
        self.on_save(self.params)
        self.destroy()


def start_audio(audio_segment):
    global play_obj
    if play_obj is not None and play_obj.is_playing():
        return
    play_obj = sa.play_buffer(
        audio_segment.raw_data,
        num_channels=audio_segment.channels,
        bytes_per_sample=audio_segment.sample_width,
        sample_rate=audio_segment.frame_rate)
    play_obj.wait_done()

class AudioEditorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Audio Editor")
        self.geometry("500x450")
        
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.init_audio_data()
        self.init_ui()

    def init_audio_data(self):
        self.original_audio = None
        self.original_audio_path = ""
        self.original_audio_samples = None
        self.modified_audio_samples = None

        self.gain_val = 0
        self.speed_diff_val = 0
        self.delay_val = 0
        self.freq_gains = np.zeros(20)
        self.filter_params = {}

    def init_ui(self):
        ctk.CTkLabel(self, text="Audio Editor", font=("Arial", 20)).pack(pady=10)

        buttons = [
            ("Wczytaj plik", self.load_file),
            ("Ustaw efekty", self.open_effects_window),
            ("Zastosuj efekty", self.apply_effects),
            ("Zapisz plik", self.save_file),
            ("▶ Odtwórz oryginalny", self.play_original_audio),
            ("▶ Odtwórz zmodyfikowany", self.play_modified_audio),
            ("|| Pauza", self.pause_audio),
            ("Zamknij", self.destroy)
        ]

        for text, command in buttons:
            ctk.CTkButton(self, text=text, command=command).pack(pady=5)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pliki audio", "*.mp3 *.wav *.flac *.m4a")])
        if not file_path:
            return

        self.original_audio_path = file_path
        ext = os.path.splitext(file_path)[-1].lower()
        self.original_audio = AudioSegment.from_file(file_path, format=ext[1:])
        self.original_audio_samples = np.array(self.original_audio.get_array_of_samples())

        if self.original_audio.channels == 2:
            self.original_audio_samples = self.original_audio_samples.reshape((-1, 2)).mean(axis=1)

        self.modified_audio_samples = self.original_audio_samples

    def play_original_audio(self):
        if self.original_audio:
            threading.Thread(target=start_audio, args=(self.original_audio,)).start()

    def pause_audio(self):
        if play_obj and play_obj.is_playing():
            play_obj.stop()

    def save_file(self):
        print("[DEBUG] Zapisz zmodyfikowany plik")

    def play_modified_audio(self):
        print("[DEBUG] Odtwarzanie zmodyfikowanego audio")

    def apply_effects(self):
        print("[DEBUG] Zastosowanie efektów do audio")

    def open_effects_window(self):
        win = ctk.CTkToplevel(self)
        win.title("Efekty audio")
        win.geometry("500x450")

        ctk.CTkLabel(win, text="Wybierz efekty i ustaw parametry", font=("Arial", 16)).pack(pady=20)

        self.gain_var = ctk.BooleanVar()
        self.echo_var = ctk.BooleanVar()
        self.reverse_var = ctk.BooleanVar()
        self.change_speed_var = ctk.BooleanVar()
        self.filter_var = ctk.BooleanVar()
        self.distortion_var = ctk.BooleanVar()
        self.noise_reduction_var = ctk.BooleanVar()
        self.delay_var = ctk.BooleanVar()

        checks = [
            ("Gain", self.gain_var, self.toggle_gain),
            ("Echo", self.echo_var, self.toggle_echo),
            ("Reverse", self.reverse_var, self.toggle_reverse),
            ("Change Speed", self.change_speed_var, self.toggle_change_speed),
            ("Filter", self.filter_var, self.toggle_filter),
            ("Distortion", self.distortion_var, self.toggle_distortion),
            ("Noise reduction", self.noise_reduction_var, self.toggle_noise_reduction),
            ("Delay", self.delay_var, self.toggle_delay)
        ]

        for text, var, cmd in checks:
            ctk.CTkCheckBox(win, text=text, variable=var, command=cmd).pack(pady=5, anchor="w", padx=30)

        ctk.CTkButton(win, text="FFT", command=self.plot_fft).pack(pady=5)
        ctk.CTkButton(win, text="Zapisz i zamknij", command=win.destroy).pack(pady=20)

    def plot_fft(self):
        if self.modified_audio_samples is None:
            return

        sample_rate = self.original_audio.frame_rate
        samples = self.modified_audio_samples.astype(np.float32) / (2 ** 15)
        N = len(samples)
        T = 1.0 / sample_rate

        yf = np.fft.fft(samples)
        xf = np.fft.fftfreq(N, T)
        idxs = np.arange(N // 2)

        xf = xf[idxs]
        yf = 2.0 / N * np.abs(yf[idxs])

        plt.figure(figsize=(10, 4))
        plt.plot(xf, yf)
        plt.title("Widmo amplitudowe (FFT)")
        plt.xlabel("Częstotliwość [Hz]")
        plt.ylabel("Amplituda")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def toggle_gain(self):
        pass
    def toggle_echo(self):
        pass
    def toggle_delay(self):
        pass
    def toggle_noise_reduction(self):
        pass
    def toggle_distortion(self):
        pass
    def toggle_change_speed(self):
        pass
    def toggle_reverse(self):
        pass

    def toggle_filter(self):
        if self.filter_var.get() == 1:
            # Otwórz okno ustawień
            FilterDesignerWindow(self, self.filter_params, self.save_filter_params)
        else:
            # Resetuj parametry
            self.filter_params = {}
            print("Filter Designer wyłączony – parametry zresetowane")
    
    def save_filter_params(self, params):
        self.filter_params = params
        print("Zapisane parametry:", self.filter_params)

if __name__ == "__main__":
    app = AudioEditorApp()
    app.mainloop()
