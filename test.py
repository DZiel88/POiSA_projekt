

import customtkinter as ctk
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
from pydub import AudioSegment
import pydub
import numpy as np
import simpleaudio as sa
import threading
from scipy.signal import freqz, firwin, iirpeak
import scipy.io
import io
# Konfiguracja pydub dla FFmpeg
# AudioSegment.converter = r"C:\\Users\\Damian\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-7.1.1-full_build\\bin\\ffmpeg.exe"
# AudioSegment.ffprobe = r"C:\\Users\\Damian\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-7.1.1-full_build\\bin\\ffprobe.exe"

# Globalna zmienna do odtwarzania
play_obj = None


class ChangeSpeedWindow(ctk.CTkToplevel):
    def __init__(self, parent, initial_params, on_save):
        super().__init__(parent)
        self.title("Ustawienia: Zmiana prędkości sygnału")
        self.geometry("200x200")
        self.on_save = on_save
        self.params = initial_params


        self.lift()
        self.focus_force()
        self.grab_set()


class NoiseReductionWindow(ctk.CTkToplevel):
    def __init__(self, parent, initial_params, on_save):
        super().__init__(parent)
        self.title("Ustawienia: Redukcja szumu sygnału")
        self.geometry("200x200")
        self.on_save = on_save
        self.params = initial_params


        self.lift()
        self.focus_force()
        self.grab_set()




class ReverseWindow(ctk.CTkToplevel):
    def __init__(self, parent, initial_params, on_save):
        super().__init__(parent)
        self.title("Ustawienia: Odwrotność sygnału")
        self.geometry("200x200")
        self.on_save = on_save
        self.params = initial_params


        self.lift()
        self.focus_force()
        self.grab_set()



class DistortionWindow(ctk.CTkToplevel):
    def __init__(self, parent, initial_params, on_save):
        super().__init__(parent)
        self.title("Ustawienia: Zniekształcenie sygnału")
        self.geometry("200x200")
        self.on_save = on_save
        self.params = initial_params

        

        self.lift()
        self.focus_force()
        self.grab_set()



class EchoWindow(ctk.CTkToplevel):
    def __init__(self, parent, initial_params, on_save):
        super().__init__(parent)
        self.title("Ustawienia: Echo sygnału")
        self.geometry("200x200")
        self.on_save = on_save
        self.params = initial_params


        self.lift()
        self.focus_force()
        self.grab_set()



class DelayWindow(ctk.CTkToplevel):
    def __init__(self, parent, initial_params, on_save):
        super().__init__(parent)
        self.title("Ustawienia: Opóźnienie sygnału")
        self.geometry("200x200")
        self.on_save = on_save
        self.params = initial_params

        self.delay_label = ctk.CTkLabel(self, text="Ustawienia opóźnienia").pack(pady=10)
        self.delay_slider = ctk.CTkSlider(self, from_=1, to=1000, number_of_steps=1000,
                                         command=self.update_freq)
        self.delay_slider.set(self.params.get("cutoff_freq", 500.0))
        self.delay_slider.pack()


        self.lift()
        self.focus_force()
        self.grab_set()



class GainWindow(ctk.CTkToplevel):
    def __init__(self, parent, initial_params, on_save):
        super().__init__(parent)
        self.title("Ustawienia: Wzmocnienie sygnału")
        self.geometry("200x200")
        self.on_save = on_save
        self.params = initial_params


        self.gain_label = ctk.CTkLabel(self, text="Ustawienia wzmocnienia").pack(pady=10)

        self.gain_entry = ctk.CTkEntry(self, placeholder_text="Wprowadź wartość wzmocnienia")
        self.gain_entry.pack(pady=20)

        self.ext_btn = ctk.CTkButton(self, text="Zapisz i wyjdź", command=self.save_and_exit).pack(pady=10)

        # Priorytetyzowanie okna
        self.lift()
        self.focus_force()
        self.grab_set()

    def update_gain(self):
        try: 
            float(self.gain_entry.get())
            if float(self.gain_entry.get()) > 5.0 or float(self.gain_entry.get()) < 0.0:
                print("Zakres wzmocnienia wynosi 0.0 - 5.0")
                return False
            else:
                self.params = float(self.gain_entry.get())
                return True
        except ValueError:
            print("[ERR] Błędny typ danych")
            return False

    def save_and_exit(self):
        self.update_gain()
        self.on_save(self.params)
        self.destroy()

class FilterDesignerWindow(ctk.CTkToplevel):
    def __init__(self, parent, initial_params, on_save, fs):
        super().__init__(parent)
        self.title("Ustawienia: Designer filtru")
        self.geometry("400x300")
        self.on_save = on_save
        self.params = initial_params.copy()
        self.freq_sampling = fs

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
        # self.q_label = ctk.CTkLabel(self, text="Q-factor: 1.0")
        # self.q_label.pack(pady=(20, 0))

        # self.q_slider = ctk.CTkSlider(self, from_=0.1, to=10.0, number_of_steps=99,
        #                               command=self.update_q)
        # self.q_slider.set(self.params.get("q_factor", 1.0))
        # self.q_slider.pack()

        # Zapisz i zamknij
        self.save_btn = ctk.CTkButton(self, text="Zapisz i zamknij", command=self.save_and_close)
        self.save_btn.pack(pady=30)

        # Priorytetyzowanie okna
        self.lift()
        self.focus_force()
        self.grab_set()


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

        self.gain_val = {}
        self.speed_coeff_val = {}
        self.echo_val = {}
        self.distortion_val = {}
        self.reverse_val = {}
        self.delay_val = {}
        self.noise_reduction_val = {}
        self.filter_params = {}

        self.effect_vars = {
            "gain":ctk.BooleanVar(value=False),
            "echo":ctk.BooleanVar(value=False),
            "reverse":ctk.BooleanVar(value=False),
            "change_speed":ctk.BooleanVar(value=False),
            "filter":ctk.BooleanVar(value=False),
            "distortion":ctk.BooleanVar(value=False),
            "noise_reduction":ctk.BooleanVar(value=False),
            "delay":ctk.BooleanVar(value=False),
        }

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

        self.sample_rate, self.original_samples = scipy.io.wavfile(str(file_path))
        
        ext = os.path.splitext(file_path)[-1].lower()
        self.original_audio = AudioSegment.from_file(file_path, format=ext[1:])
        self.original_audio_samples = np.array(self.original_audio.get_array_of_samples())

        if self.original_audio.channels == 2:
            self.original_audio_samples = self.original_audio_samples.reshape((-1, 2)).mean(axis=1)

        self.modified_audio_samples = self.original_audio_samples
        self.max_val = np.max(np.abs(self.modified_audio_samples))

    def play_original_audio(self):
        if self.original_audio:
            threading.Thread(target=start_audio, args=(self.original_audio,)).start()

    def pause_audio(self):
        if play_obj and play_obj.is_playing():
            play_obj.stop()

    def save_file(self):
        for sample in self.original_audio_samples:
            print(sample)

        self.original_audio = audio_segment = AudioSegment.from_raw(
        io.BytesIO(self.original_audio_samples),
        sample_width=2,
        frame_rate=44100,
        channels=1
)
        print(self.original_audio_samples)
        print(self.original_audio_samples.dtype)
        print(self.original_audio_samples.size)

    def play_modified_audio(self):
        if self.modified_audio_segment is not None:
            threading.Thread(target=start_audio, args=(self.modified_audio_segment,)).start()

    def apply_effects(self):
        
        checks = [
            ("Gain", self.effect_vars["gain"], self.apply_gain),
            ("Echo", self.effect_vars["echo"], self.apply_echo),
            ("Reverse", self.effect_vars["reverse"], self.apply_reverse),
            ("Change Speed", self.effect_vars["change_speed"], self.apply_change_speed),
            ("Filter", self.effect_vars["filter"], self.apply_filter),
            ("Distortion", self.effect_vars["distortion"], self.apply_distortion),
            ("Noise reduction", self.effect_vars["noise_reduction"], self.apply_noise_reduction),
            ("Delay", self.effect_vars["delay"], self.apply_delay)
        ]
        for effect, var, cmd in checks:
            if var.get() == True:
                cmd()
                # print(effect + " applied")

        self.modified_audio_segment = AudioSegment(
            self.modified_audio_samples,
            frame_rate=self.original_audio.frame_rate,
            sample_width=2,
            channels=1
        )
        
        

    def open_effects_window(self):
        self.win = ctk.CTkToplevel(self)
        self.win.title("Efekty audio")
        self.win.geometry("500x450")

        ctk.CTkLabel(self.win, text="Wybierz efekty i ustaw parametry", font=("Arial", 16)).pack(pady=20)


        checks = [
            ("Gain", self.effect_vars["gain"], self.toggle_gain),
            ("Echo", self.effect_vars["echo"], self.toggle_echo),
            ("Reverse", self.effect_vars["reverse"], self.toggle_reverse),
            ("Change Speed", self.effect_vars["change_speed"], self.toggle_change_speed),
            ("Filter", self.effect_vars["filter"], self.toggle_filter),
            ("Distortion", self.effect_vars["distortion"], self.toggle_distortion),
            ("Noise reduction", self.effect_vars["noise_reduction"], self.toggle_noise_reduction),
            ("Delay", self.effect_vars["delay"], self.toggle_delay)
        ]

        for text, var, cmd in checks:
            ctk.CTkCheckBox(self.win, text=text, variable=var, command=cmd).pack(pady=5, anchor="w", padx=30)

        ctk.CTkButton(self.win, text="FFT", command=self.plot_fft).pack(pady=5)
        ctk.CTkButton(self.win, text="Zapisz i zamknij", command=self.win.destroy).pack(pady=20)

         # Priorytetyzowanie okna
        self.win.lift()
        self.win.focus_force()
        self.win.grab_set()

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
        if self.effect_vars["gain"].get() == 1:
            # Otwórz okno ustawień
            GainWindow(self, initial_params=self.gain_val, on_save=self.save_gain)
        else:
            # Resetuj parametry
            self.filter_params = {}
            print("Gain wyłączony – parametry zresetowane")


    def toggle_echo(self):
        if self.effect_vars["echo"].get() == 1:
            # Otwórz okno ustawień
            EchoWindow(self, initial_params=self.echo_val, on_save=self.save_echo)
        else:
            # Resetuj parametry
            self.echo_val = {}
            print("Filter Designer wyłączony – parametry zresetowane")
        
    def toggle_delay(self):
        if self.effect_vars["delay"].get() == 1:
            # Otwórz okno ustawień
            DelayWindow(self, initial_params=self.delay_val, on_save=self.save_delay)
        else:
            # Resetuj parametry
            self.delay_val = {}
            print("Filter Designer wyłączony – parametry zresetowane")

    def toggle_noise_reduction(self):
        if self.effect_vars["noise_reduction"].get() == 1:
            # Otwórz okno ustawień
            NoiseReductionWindow(self, initial_params=self.noise_reduction_val, on_save=self.save_noise_reduction)
        else:
            # Resetuj parametry
            self.delay_val = {}
            print("Filter Designer wyłączony – parametry zresetowane")

    def toggle_distortion(self):
        if self.effect_vars["distortion"].get() == 1:
            # Otwórz okno ustawień
            DistortionWindow(self, initial_params=self.distortion_val, on_save=self.save_distortion)
        else:
            # Resetuj parametry
            self.distortion_val = {}
            print("Filter Designer wyłączony – parametry zresetowane")

    def toggle_change_speed(self):
        if self.effect_vars["change_speed"].get() == 1:
            # Otwórz okno ustawień
            ChangeSpeedWindow(self, initial_params=self.speed_coeff_val, on_save=self.save_change_speed)
        else:
            # Resetuj parametry
            self.speed_coeff_val = {}
            print("Filter Designer wyłączony – parametry zresetowane")

    def toggle_reverse(self):
        if self.effect_vars["reverse"].get() == 1:
            # Otwórz okno ustawień
            ReverseWindow(self, initial_params=self.reverse_val, on_save=self.save_reverse)
        else:
            # Resetuj parametry
            self.reverse_val = {}
            print("Filter Designer wyłączony – parametry zresetowane")

    def toggle_filter(self):
        if self.effect_vars["filter"].get() == 1:
            # Otwórz okno ustawień
            FilterDesignerWindow(self, self.filter_params, self.save_filter_params, self.original_audio.frame_rate)
        else:
            # Resetuj parametry
            self.filter_params = {}
            print("Filter Designer wyłączony – parametry zresetowane")
    
    def save_gain(self, params):
        self.gain_val = params
        print("Zapisana wartość wzmocnienia ", self.gain_val)

    def save_delay(self, params):
        self.delay_val = params
        print("Zapisana wartość opóźnienia to ", self.delay_val + "[ms]")

    def save_echo(self, params):
        self.echo_val = params
        print("Zapisana wartość echa", self.echo_val)

    def save_distortion(self, params):
        self.distortion_val = params
        print("Zapisana wartość zniekształcenia", self.distortion_val)

    def save_noise_reduction(self, params):
        self.noise_reduction_val = params
        print("Zapisana wartość tłumienia", self.noise_reduction_val)

    def save_change_speed(self, params):
        self.speed_coeff_val = params
        print("Zapisana wartość tłumienia", self.noise_reduction_val)

    def save_reverse(self, params):
        self.reverse_val = params
        print("Zapisana wartość odwrotnosci", self.reverse_val)

    def save_filter_params(self, params):
        self.filter_params = params
        print("Zapisane parametry:", self.filter_params)

    def apply_gain(self):
        # self.modified_audio_segment.apply_gain(self.gain_val)
        # self.modified_audio_samples *= self.gain_val
        # print(self.modified_audio_samples[:])
        pass
        


    def apply_echo(self):
        print("echo applied")
    
    def apply_delay(self):
        print("delay applied")

    def apply_noise_reduction(self):
        print("noise reduction applied")

    def apply_distortion(self):
        print("distortion applied")

    def apply_change_speed(self):
        print("change speed applied")

    def apply_reverse(self):
        print("reverse applied")

    def apply_filter(self):
        print("filter applied")


if __name__ == "__main__":
    app = AudioEditorApp()
    app.mainloop()
