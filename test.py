
import customtkinter as ctk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import threading
from scipy.signal import freqz, firwin, iirpeak, butter, lfilter
import scipy.io
import sounddevice


class ChangeSpeedWindow(ctk.CTkToplevel):
    def __init__(self, parent, initial_params, on_save):
        super().__init__(parent)
        self.title("Ustawienia: Zmiana prędkości sygnału")
        self.geometry("250x130")
        self.on_save = on_save
        self.params = initial_params

        self.speed_label = ctk.CTkLabel(self, text="Współczynnik zmiany szybkości: 1.0")
        self.speed_label.pack(pady=(10, 0))

        self.gain_slider = ctk.CTkSlider(self, from_=0, to=4, number_of_steps=40,
                                         command=self.update_speed)
        
        self.gain_slider.set(1)
        self.gain_slider.pack()

        self.ext_btn = ctk.CTkButton(self, text="Zapisz i wyjdź", command=self.save_and_exit).pack(pady=10)

        self.lift()
        self.focus_force()
        self.grab_set()
    
    def update_speed(self, factor):
        self.params = factor
        self.speed_label.configure(text=f"Współczynnik zmiany szybkości: {factor:.1f}")

    def save_and_exit(self):
        self.on_save(self.params)
        self.destroy()



# class NoiseReductionWindow(ctk.CTkToplevel):
#     def __init__(self, parent, initial_params, on_save):
#         super().__init__(parent)
#         self.title("Ustawienia: Redukcja szumu sygnału")
#         self.geometry("200x200")
#         self.on_save = on_save
#         self.params = initial_params


#         self.lift()
#         self.focus_force()
#         self.grab_set()




class ReverseWindow(ctk.CTkToplevel):
    def __init__(self, parent, initial_params, on_save):
        super().__init__(parent)
        self.title("Ustawienia: Odwrotność sygnału")
        self.geometry("200x200")
        self.on_save = on_save
        self.params = initial_params

        self.if_checked = ctk.BooleanVar(value=False)

        self.label = ctk.CTkLabel(self, text="Odwrócenie sygnału")
        self.label.pack(pady=20)

        self.reverse_checkbox = ctk.CTkCheckBox(self, text="Odwróć sygnał", variable=self.if_checked, command=self.update_reverse).pack(pady=10)

        self.ext_btn = ctk.CTkButton(self, text="Zapisz i wyjdź", command=self.save_and_exit).pack(pady=10)

        self.lift()
        self.focus_force()
        self.grab_set()
    
    def update_reverse(self):
        self.params = True

    def save_and_exit(self):
        self.on_save(self.params)
        self.destroy()
        



class DistortionWindow(ctk.CTkToplevel):
    def __init__(self, parent, initial_params, on_save):
        super().__init__(parent)
        self.title("Ustawienia: Zniekształcenie sygnału")
        self.geometry("300x230")
        self.on_save = on_save
        self.params = initial_params
        self.params["dist_type"] = ""
        self.params["dist_gain"] = 0 # w dB
        self.params["dist_th"] = 0

        self.distortion_type_label = ctk.CTkLabel(self, text="Typ zniekształcenia: Łagodne obcięcie")
        self.distortion_type_label.pack(pady=(10, 0))

        self.distortion_type = ctk.CTkOptionMenu(
            self,
            values=["Twarde obcięcie", "Łagodne obcięcie", "Zniekształcenie wykładnicze", "Zniekształcenie artan"],
            command=self.update_distortion_type
        )

        self.distortion_type.set("Łagodne obcięcie")
        self.distortion_type.pack()


        self.distortion_gain_label = ctk.CTkLabel(self, text="Wzmocnienie zniekształcenia: 0.0 dB (1.000x)")
        self.distortion_gain_label.pack(pady=(10, 0))

        self.gain_slider = ctk.CTkSlider(self, from_=-18, to=18, number_of_steps=72,
                                         command=self.update_distortion_gain)
        
        self.gain_slider.set(0)
        self.gain_slider.pack()

        self.distortion_th_label = ctk.CTkLabel(self, text="Próg: 0.5")
        self.distortion_th_label.pack(pady=(10, 0))

        self.th_slider = ctk.CTkSlider(self, from_=0, to=1, number_of_steps=100,
                                         command=self.update_distortion_threshold)
        
        self.th_slider.set(0.5)
        self.th_slider.pack()

        self.ext_btn = ctk.CTkButton(self, text="Zapisz i wyjdź", command=self.save_and_exit).pack(pady=10)

        self.lift()
        self.focus_force()
        self.grab_set()


    def update_distortion_type(self, type):
        self.params["dist_type"] = type
        self.distortion_type_label.configure(text=f"Typ zniekształcenia {str(type)}")

    def update_distortion_gain(self, gain):
        self.gain_db = round(gain, 2)
        self.params["dist_gain"] = self.update_gain(gain=gain)
        self.distortion_gain_label.configure(text=f"Wzmocnienie: {self.gain_db:.1f} dB ({self.update_gain(gain=gain):.3f}x)")

    def update_distortion_threshold(self, th):
        
        self.params["dist_th"] = th
        self.distortion_th_label.configure(text=f"Próg: {th:.2f}")  
        
    def update_gain(self, gain):
        return 10**(gain/20)

    def save_and_exit(self):
        self.on_save(self.params)
        self.destroy()



class EchoWindow(ctk.CTkToplevel):
    def __init__(self, parent, initial_params, on_save):
        super().__init__(parent)
        self.title("Ustawienia: Echo sygnału")
        self.geometry("250x300")
        self.on_save = on_save
        self.params = initial_params
        self.params["echo_delay"] = 0
        self.params["echo_decay"] = 0
        self.params["echo_iterations"] = 5

        self.label_delay = ctk.CTkLabel(self, text="Opóźnienie echa: 1250 [ms]")
        self.label_delay.pack(pady=10)

        self.delay_slider = ctk.CTkSlider(self, from_=0, to=2500, number_of_steps=250,
                                         command=self.update_echo_delay)
        self.delay_slider.set(1250)
        self.delay_slider.pack()



        self.label_decay = ctk.CTkLabel(self, text="Współczynnik tłumienia: 0.5")
        self.label_decay.pack(pady=10)
        self.decay_slider = ctk.CTkSlider(self, from_=0, to=1, number_of_steps=100,
                                         command=self.update_echo_decay)
        
        self.decay_slider.set(0.5)
        self.decay_slider.pack()


        self.label_iterations = ctk.CTkLabel(self, text="Liczba powtórzeń sygnału: 5")
        self.label_iterations.pack(pady=10)
        self.iterations_slider = ctk.CTkSlider(self, from_=0, to=10, number_of_steps=10,
                                         command=self.update_echo_iterations)
        
        self.iterations_slider.set(5)
        self.iterations_slider.pack()

        self.ext_btn = ctk.CTkButton(self, text="Zapisz i wyjdź", command=self.save_and_exit).pack(pady=10)

        self.lift()
        self.focus_force()
        self.grab_set()

    def update_echo_delay(self, delay):
        self.params["echo_delay"] = int(delay)
        self.label_delay.configure(text=f"Opóźnienie echa: {delay:.0f} ms")

    def update_echo_decay(self, decay):
        self.params["echo_decay"] = round(decay, 2)
        self.label_decay.configure(text=f"Współczynnik tłumienia: {decay:.2f}")

    def update_echo_iterations(self, iterations):
        self.params["echo_iterations"] = int(iterations)
        self.label_iterations.configure(text=f"Liczba powtórzeń sygnału: {iterations:.0f}")
    
    def save_and_exit(self):
        self.on_save(self.params)
        self.destroy()



class DelayWindow(ctk.CTkToplevel):
    def __init__(self, parent, initial_params, on_save):
        super().__init__(parent)
        self.title("Ustawienia: Opóźnienie sygnału")
        self.geometry("200x200")
        self.on_save = on_save
        self.params = initial_params

        self.label = ctk.CTkLabel(self, text="Ustawienia opóźnienia [ms]")
        self.label.pack(pady=10)
        self.delay_slider = ctk.CTkSlider(self, from_=0, to=5000, number_of_steps=500,
                                         command=self.update_delay)
        self.delay_slider.set(2500)
        self.delay_slider.pack()

        self.ext_btn = ctk.CTkButton(self, text="Zapisz i wyjdź", command=self.save_and_exit).pack(pady=10)

        self.lift()
        self.focus_force()
        self.grab_set()
    
    def update_delay(self, value):
        self.params = int(value)
        self.label.configure(text=f"Opóźnienie: {self.params:.0f} ms")

    def save_and_exit(self):
        self.on_save(self.params)
        self.destroy()



class GainWindow(ctk.CTkToplevel):
    def __init__(self, parent, initial_params, on_save):
        super().__init__(parent)
        self.title("Ustawienia: Wzmocnienie sygnału")
        self.geometry("200x200")
        self.on_save = on_save
        self.params = initial_params


        self.label = ctk.CTkLabel(self, text="Wzmocnienie: 0.0 dB (1.000x)")
        self.label.pack(pady=20)

        self.gain_slider = ctk.CTkSlider(
            self,
            from_=-18,
            to=18,
            number_of_steps=72,
            command=self.gain_slider_changed
        )
        self.gain_slider.set(0)
        self.gain_slider.pack(pady=10)

        self.ext_btn = ctk.CTkButton(self, text="Zapisz i wyjdź", command=self.save_and_exit).pack(pady=10)

        # Priorytetyzowanie okna
        self.lift()
        self.focus_force()
        self.grab_set()
        

    def gain_slider_changed(self, value):
        self.gain_db = round(value, 2)
        self.params = self.update_gain(self.gain_db)
        self.label.configure(text=f"Wzmocnienie: {self.gain_db:.1f} dB ({self.params:.3f}x)")

    def update_gain(self, gain):
        return 10**(gain/20)

    def save_and_exit(self):
        self.on_save(self.params)
        self.destroy()

class FilterDesignerWindow(ctk.CTkToplevel):
    def __init__(self, parent, initial_params, on_save):
        super().__init__(parent)
        self.title("Ustawienia: Designer filtru")
        self.geometry("500x450")
        self.on_save = on_save
        self.params = initial_params.copy()
        self.params["filter_type"] = "lowpass"
        self.params["filter_order"] = 60
        self.params["filter_method"] = "FIR"
        self.params["cutoff_freq1"] = 500
        self.params["cutoff_freq2"] = 20000

        # Typ filtra
        self.filter_type_label = ctk.CTkLabel(self, text="Typ filtra: ")
        self.filter_type_label.pack(pady=(10, 0))

        self.filter_type = ctk.CTkOptionMenu(
            self,
            values=["lowpass", "highpass", "bandpass"],
            command=self.update_filter_type
        )
        self.filter_type.set("lowpass")
        self.filter_type.pack()

        self.filter_method_label = ctk.CTkLabel(self, text="Metoda filtru: ")
        self.filter_method_label.pack(pady=(10, 0))

        self.filter_method = ctk.CTkOptionMenu(
            self,
            values=["FIR", "Czebyszew", "Butterworth"],
            command=self.update_filter_method
        )
        self.filter_method.set("FIR")
        self.filter_method.pack()



        # Częstotliwość graniczna
        self.freqf1_label = ctk.CTkLabel(self, text="Częstotliwość graniczna f1: 500 Hz")
        self.freqf1_label.pack(pady=(20, 0))

        self.freqf1_slider = ctk.CTkSlider(self, from_=0, to=20000, number_of_steps=20000,
                                         command=self.update_freqf1)
        self.freqf1_slider.set(500.0)
        self.freqf1_slider.pack()

        self.freqf2_label = ctk.CTkLabel(self, text="(Dla typu bandpass) Częstotliwość graniczna f2: 20000 Hz")
        self.freqf2_label.pack(pady=(20, 0))

        self.freqf2_slider = ctk.CTkSlider(self, from_=0, to=20000, number_of_steps=20000,
                                         command=self.update_freqf2)
        self.freqf2_slider.set(20000)
        self.freqf2_slider.pack()



        self.order_label = ctk.CTkLabel(self, text="Rząd filtru: 60")
        self.order_label.pack(pady=(20, 0))

        self.order_slider = ctk.CTkSlider(self, from_=1, to=120, number_of_steps=119,
                                         command=self.update_filter_order)
        self.order_slider.set(60)
        self.order_slider.pack()


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

    def update_filter_order(self, val):
        self.params["filter_order"] = int(val)
        self.order_label.configure(text=f"Rząd filtru: {int(val):.0f}")

    def update_filter_method(self, val):
        self.params["filter_method"] = val

    def update_freqf1(self, val):
        val = round(val)
        self.freqf1_label.configure(text=f"Częstotliwość graniczna f1: {val} Hz")
        self.params["cutoff_freq1"] = val

    def update_freqf2(self, val):
        val = round(val)
        self.freqf2_label.configure(text=f"(Dla typu bandpass) Częstotliwość graniczna f2: {val} Hz")
        self.params["cutoff_freq2"] = val

    def save_and_close(self):
        if self.params["cutoff_freq1"] > self.params["cutoff_freq2"]:
            print("Cutoff_1 musi być mniejsza od cutoff_2")
            return
        self.on_save(self.params)
        self.destroy()


def start_audio(audio_segment, rate):
    print("playing")
    sounddevice.play(audio_segment, samplerate=rate)
    # sounddevice.wait()

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
        self.original_samples = None
        self.modified_audio_samples = None

        self.gain_val = {}
        self.speed_coeff_val = {}
        self.echo_val = {}
        self.distortion_val = {}
        self.reverse_val = {}
        self.delay_val = {}
        # self.noise_reduction_val = {}
        self.filter_params = {}
        self.audio_thread = {}

        self.effect_vars = {
            "reverse":ctk.BooleanVar(value=False),
            "gain":ctk.BooleanVar(value=False),
            "echo":ctk.BooleanVar(value=False),
            "change_speed":ctk.BooleanVar(value=False),
            "filter":ctk.BooleanVar(value=False),
            "distortion":ctk.BooleanVar(value=False),
            # "noise_reduction":ctk.BooleanVar(value=False),
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
            ("Reset zmodyfikowanego sygnału", self.reset),
            ("FFT zmodulowanego sygnału", self.plot_fft),
            ("Zamknij", self.destroy)
            
        ]

        for text, command in buttons:
            ctk.CTkButton(self, text=text, command=command).pack(pady=5)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pliki audio", "*.wav")])
        if not file_path:
            return

        (self.sample_rate, self.original_samples) = scipy.io.wavfile.read(str(file_path))
        self.original_samples = self.original_samples.astype(np.float32) / np.iinfo(np.int16).max # Do float32 

        if len(self.original_samples.shape) == 2:
            self.original_samples = self.original_samples.mean(axis=1)

        self.modified_audio_samples = self.original_samples

    def play_original_audio(self):
        if self.original_samples is not None:
            self.audio_thread = threading.Thread(target=start_audio, args=(self.original_samples, self.sample_rate))
            self.audio_thread.start()

    def play_modified_audio(self):
        if self.modified_audio_samples is not None:
            self.audio_thread = threading.Thread(target=start_audio, args=(self.modified_audio_samples, self.sample_rate))
            self.audio_thread.start()

    def pause_audio(self):
        sounddevice.stop()
        if self.audio_thread is not None and self.audio_thread.is_alive():
            self.audio_thread.join()

    def reset(self):
        self.modified_audio_samples = self.original_samples


    def save_file(self):
        if self.modified_audio_samples.dtype == np.float32 or self.modified_audio_samples.dtype == np.float64:
            self.modified_audio_samples = np.clip(self.modified_audio_samples, -1.0, 1.0)  # zabezpieczenie
            self.modified_audio_samples = (self.modified_audio_samples * 32767).astype(np.int16)
        scipy.io.wavfile.write("output.wav", self.sample_rate, self.modified_audio_samples)
        # print(self.original_samples)
        

    

    def apply_effects(self):
        
        checks = [
            ("Reverse", self.effect_vars["reverse"], self.apply_reverse),
            ("Gain", self.effect_vars["gain"], self.apply_gain),
            ("Echo", self.effect_vars["echo"], self.apply_echo),
            ("Change Speed", self.effect_vars["change_speed"], self.apply_change_speed),
            ("Filter", self.effect_vars["filter"], self.apply_filter),
            ("Distortion", self.effect_vars["distortion"], self.apply_distortion),
            ("Delay", self.effect_vars["delay"], self.apply_delay)
        ]
        for effect, var, cmd in checks:
            if var.get() == True:
                cmd()
                # print(effect + " applied")

        
        

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
            ("Delay", self.effect_vars["delay"], self.toggle_delay)
        ]

        for text, var, cmd in checks:
            ctk.CTkCheckBox(self.win, text=text, variable=var, command=cmd).pack(pady=5, anchor="w", padx=30)

        ctk.CTkButton(self.win, text="Zapisz i zamknij", command=self.win.destroy).pack(pady=20)

         # Priorytetyzowanie okna
        self.win.lift()
        self.win.focus_force()
        self.win.grab_set()

    def plot_fft(self):

        if self.modified_audio_samples is None:
            return
        
        samples = self.modified_audio_samples.astype(np.float32) / (2 ** 15)
        N = len(samples)
        T = 1.0 / self.sample_rate

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

    # def toggle_noise_reduction(self):
    #     if self.effect_vars["noise_reduction"].get() == 1:
    #         # Otwórz okno ustawień
    #         NoiseReductionWindow(self, initial_params=self.noise_reduction_val, on_save=self.save_noise_reduction)
    #     else:
    #         # Resetuj parametry
    #         self.delay_val = {}
    #         print("Filter Designer wyłączony – parametry zresetowane")

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
            FilterDesignerWindow(self, self.filter_params, self.save_filter_params)
        else:
            # Resetuj parametry
            self.filter_params = {}
            print("Filter Designer wyłączony – parametry zresetowane")
    
    def save_gain(self, params):
        self.gain_val = params
        print("Zapisana wartość wzmocnienia ", self.gain_val)

    def save_delay(self, params):
        self.delay_val = params
        print("Zapisana wartość opóźnienia to ", self.delay_val, "[ms]")

    def save_echo(self, params):
        self.echo_val = params
        print("Zapisane parametry echa ", self.echo_val)

    def save_distortion(self, params):
        self.distortion_val = params
        print("Zapisana wartość zniekształcenia ", self.distortion_val)

    # def save_noise_reduction(self, params):
    #     self.noise_reduction_val = params
    #     print("Zapisana wartość tłumienia", self.noise_reduction_val)

    def save_change_speed(self, params):
        self.speed_coeff_val = params
        print("Zapisana wartość współczynnika prędkości", self.speed_coeff_val)

    def save_reverse(self, params):
        self.reverse_val = params
        print("Zapisana wartość odwrotnosci", self.reverse_val)

    def save_filter_params(self, params):
        self.filter_params = params
        print("Zapisane parametry:", self.filter_params)





    def apply_gain(self):
        print("gain applied")
        self.modified_audio_samples = self.modified_audio_samples * self.gain_val
        
    def apply_echo(self):
        delay_samples = int(self.echo_val["echo_delay"]/1000 * self.sample_rate)
        output_length = int(len(self.modified_audio_samples) + delay_samples * self.echo_val["echo_iterations"])
        output = np.zeros(output_length)

    # Dodaj oryginalny sygnał
        output[:len(self.modified_audio_samples)] += self.modified_audio_samples

    # Dodaj każde echo
        for i in range(1, self.echo_val["echo_iterations"] + 1):
            start = delay_samples * i
            end = start + len(self.modified_audio_samples)
            if end > output_length:
                end = output_length
            echo_signal = self.echo_val["echo_decay"]**i * self.modified_audio_samples[:end - start]
            output[start:end] += echo_signal
        self.modified_audio_samples = output
        print("echo applied")
        
    
    def apply_delay(self):
        self.modified_audio_samples = np.concatenate((np.zeros(round(self.sample_rate/1000)*self.delay_val), self.modified_audio_samples))
        print("delay applied")

    # def apply_noise_reduction(self):
    #     print("noise reduction applied")

    def apply_distortion(self):
        if self.distortion_val["dist_type"] == "Łagodne obcięcie":
            self.modified_audio_samples = np.tanh(self.modified_audio_samples*self.distortion_val["dist_gain"])
        elif self.distortion_val["dist_type"] == "Zniekształcenie wykładnicze":
            self.modified_audio_samples = np.sign(self.modified_audio_samples) * (1 - np.exp(-np.abs(self.distortion_val["dist_gain"] * self.modified_audio_samples)))
        elif self.distortion_val["dist_type"] == "Twarde obcięcie":
            self.modified_audio_samples = np.clip(self.modified_audio_samples, -self.distortion_val["dist_th"], self.distortion_val["dist_th"])
        else:
            self.modified_audio_samples = (2 / np.pi) * np.arctan(self.distortion_val["dist_gain"] * self.modified_audio_samples)

        self.modified_audio_samples /= np.max(np.abs(self.modified_audio_samples) + 1e-9)

        print("distortion applied")

    def apply_change_speed(self):
        n_samples = int(len(self.modified_audio_samples) / self.speed_coeff_val)
        self.modified_audio_samples = scipy.signal.resample(self.modified_audio_samples, n_samples)
        print("change speed applied")

    def apply_reverse(self):
        self.modified_audio_samples = self.modified_audio_samples[::-1]
        print("reverse applied")

    def apply_filter(self):
        if self.filter_params["filter_method"] == "Butterworth":
            self.modified_audio_samples = self.apply_Butterworth_filter(
                cutoff_hz1=self.filter_params["cutoff_freq1"],
                cutoff_hz2=self.filter_params["cutoff_freq2"],
                filter_type=self.filter_params["filter_type"],
                order=self.filter_params["filter_order"]
            )
        elif self.filter_params["filter_method"] == "FIR":
            self.modified_audio_samples = self.apply_fir_filter(
                cutoff_hz1=self.filter_params["cutoff_freq1"],
                cutoff_hz2=self.filter_params["cutoff_freq2"],
                filter_type=self.filter_params["filter_type"],
                order=self.filter_params["filter_order"]
            )
        elif self.filter_params["filter_method"] == "Czebyszew":
            self.modified_audio_samples = self.apply_Czebyszew_filter(
                cutoff_hz1=self.filter_params["cutoff_freq1"],
                cutoff_hz2=self.filter_params["cutoff_freq2"],
                filter_type=self.filter_params["filter_type"],
                order=self.filter_params["filter_order"]
            )
        print("filter applied")
    

    def apply_Butterworth_filter(self, cutoff_hz1, cutoff_hz2, filter_type="lowpass", order=4):
        nyq = 0.5 * self.sample_rate
        if filter_type in ("lowpass", "highpass"):
            cutoff = cutoff_hz1 / nyq
        elif filter_type == "bandpass":
            cutoff = [cutoff_hz1 / nyq, cutoff_hz2 / nyq]
        else:
            raise ValueError("Invalid filter_type")

        b, a = butter(order, cutoff, btype=filter_type)
        filtered = lfilter(b, a, self.modified_audio_samples)
        
        return filtered

    def apply_fir_filter(self, cutoff_hz1, cutoff_hz2, filter_type="lowpass", order=101):
        nyq = 0.5 * self.sample_rate
        if filter_type in ("lowpass", "highpass"):
            cutoff = cutoff_hz1 / nyq
        elif filter_type == "bandpass":
            cutoff = [cutoff_hz1 / nyq, cutoff_hz2 / nyq]  # cutoff_hz = [low, high]
        else:
            raise ValueError("Invalid filter_type")

        fir_coeff = firwin(order, cutoff, pass_zero=filter_type)
        filtered = lfilter(fir_coeff, 1.0, self.modified_audio_samples)
        return filtered
    
    def apply_Czebyszew_filter(self, cutoff_hz1, cutoff_hz2=None, filter_type="lowpass", order=4, rp=1):
    
        nyq = 0.5 * self.sample_rate

        if filter_type in ("lowpass", "highpass"):
            cutoff = cutoff_hz1 / nyq
        elif filter_type == "bandpass":
            if cutoff_hz2 is None:
                raise ValueError("Bandpass/bandstop requires two cutoff frequencies.")
            cutoff = [cutoff_hz1 / nyq, cutoff_hz2 / nyq]
        else:
            raise ValueError("Invalid filter_type")

        b, a = scipy.signal.cheby1(order, rp, cutoff, btype=filter_type)
        filtered = lfilter(b, a, self.modified_audio_samples)
        return filtered

if __name__ == "__main__":
    app = AudioEditorApp()
    app.mainloop()
