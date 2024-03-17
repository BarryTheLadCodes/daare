#!/usr/bin/env python
import soundfile as sf
import pyloudnorm as ln
from spleeter.separator import Separator
import numpy as np
import librosa
import argparse 
import sys, os

# LOUDNESS AND DYNAMICS
class LND:
    def __init__(self, path):
        self.path = path
        self.audio, self.rate = sf.read(path) #load audio and sampling rate
        self.meter = ln.Meter(rate=self.rate) #create loudness meter for given sampling rate
        self.loudness = round(self.meter.integrated_loudness(self.audio),1)
        # compute dBTP value to avoid clipping the dialog with amplification

def dBFS(input_RMS):
    return 20*np.log10( input_RMS )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dialog-aware Remixing Engine for Better Dialog Clarity in Stereo Audio')
    parser.add_argument('-s',  dest='source', help='required: </path/to/asset_2p0_source.wav> path to stereo audio source file', required=True, type=str)
    parser.add_argument('-b',  dest='boost', help='required: level of dialog boost in dBFS e.g. 1, 3, 6 etc.', required=True, type=int)
    parser.add_argument('-v',  dest='vocals', help='optional: </path/to/asset_2p0_vocals.wav> path to extracted vocals', required=False, type=str)
    parser.add_argument('-r',  dest='residual', help='optional: </path/to/asset_2p0_residual.wav> path to extracted residual', required=False, type=str)
    parser.add_argument('-o',  dest='output', help='required: </path/to/asset_2p0_daare.wav> path to DAARE output', required=True, type=str)
    try:
        args = parser.parse_args()
    except SystemExit as e:
        print(e)
        sys.exit(1)
    if len( sys.argv ) < 3:
        print('Not enough required input arguments were provided.')
        sys.exit(1)

    source_path = args.source
    boost_dBFS = args.boost
    output_path = args.output
    filename = os.path.basename(os.path.splitext(source_path)[0])

    output_vocal_path = os.path.join(output_path,filename+"_vocals.wav")
    output_residual_path = os.path.join(output_path,filename+"_residual.wav")
    output_vocalboost_path = os.path.join(output_path,filename+"_vocals_boost.wav")
    output_residualattn_path = os.path.join(output_path,filename+"_residual_attenuated.wav")
    output_daare_path = os.path.join(output_path,filename+"_DAARE_"+str(boost_dBFS)+"dB.wav")

    sample_rate = librosa.get_samplerate(source_path)    
    source,_ = sf.read(source_path)

    if (args.vocals):
         print("SPLEETER 2stems-16kHz outputs already available!")
         save_spleeter_outputs = False   
    else:
        print("SPLEETER 2stems-16kHz model running...")
        separator = Separator('spleeter:2stems-16kHz') #create spleeter separator object using 2stems model (vocals and residual)    
        seperated_audio = separator.separate(source) #separate audio using the separator object into two tracks
        vocals = seperated_audio['vocals']
        residual = seperated_audio['accompaniment']
        sf.write( output_vocal_path, vocals, sample_rate)
        sf.write( output_residual_path, residual, sample_rate)
        save_spleeter_outputs = True
    
    # Downmix to mono to preserve the stereo image of the mix
    source,sample_rate = librosa.load(source_path,mono=False,sr=sample_rate)
    vocals,sample_rate = librosa.load(output_vocal_path,mono=False,sr=sample_rate)
    residual,sample_rate = librosa.load(output_residual_path,mono=False,sr=sample_rate)
    source_mono = librosa.to_mono(source)
    vocals_mono = librosa.to_mono(vocals)
    residual_mono = librosa.to_mono(residual)

    # Compute Root Mean Square of Source, Vocals and Residual
    # next -> replace averages by chunked avg RMS values
    
    rms_s = np.average(librosa.feature.rms(y=source_mono))
    rms_v = np.average(librosa.feature.rms(y=vocals_mono))
    rms_r = np.average(librosa.feature.rms(y=residual_mono))

    # Attenuate residuals
    attn_gain = ( 10**(-boost_dBFS/20) ) * np.sqrt( rms_r / rms_s )
    residual_attn = attn_gain * residual
    # Amplify vocals
    amp_gain = ( 10**(boost_dBFS/20) ) * np.sqrt( rms_v / rms_s )
    vocals_amp = amp_gain * vocals
  
    print("-----------------------------------------------------")
    print("Source = ",dBFS(rms_s), "dBFS")
    print("Vocals = ",dBFS(rms_v), "dBFS")
    print("Amplified Vocals = ",dBFS( np.average(librosa.feature.rms(y=vocals_amp)) ), "dBFS")
    print("Residual = ",dBFS(rms_r), "dBFS")
    print("Attenuated Residual = ",dBFS( np.average(librosa.feature.rms(y=residual_attn)) ), "dBFS")
    print("Soure-to-Vocals = ",   dBFS(rms_s / rms_v), "dBFS")
    print("Source-to-Residual = ",dBFS(rms_s / rms_r), "dBFS")
    print("Vocals-to-Residual = ",dBFS(rms_v / rms_r), "dBFS")
    print("Amplification gain for vocals = ", amp_gain)
    print("Attenuation gain for residual = ", attn_gain)   

    # DAARE outputs 
    reconstructed = vocals + residual
    # daare_output = vocals + residual * 0.25
    daare_output = vocals_amp + residual_attn

    if (save_spleeter_outputs==False):
        sf.write( output_vocal_path, np.transpose(vocals), sample_rate)
        sf.write( output_residual_path, np.transpose(residual), sample_rate)
    sf.write( output_vocalboost_path, np.transpose(vocals_amp), sample_rate)
    sf.write( output_residualattn_path, np.transpose(residual_attn), sample_rate)
    sf.write( output_daare_path, np.transpose(daare_output), sample_rate)

    print("-----------------------------------------------------")
    print(f"Source : {LND(source_path).loudness} LKFS")
    print(f"Spleeter Vocals : {LND(output_vocal_path).loudness} LKFS")
    print(f"Amplified Vocals : {LND(output_vocalboost_path).loudness} LKFS")
    print(f"Spleeter Residual : {LND(output_residual_path).loudness} LKFS")
    print(f"Attenuated Residual : {LND(output_residualattn_path).loudness} LKFS")
    print(f"DAARE output : {LND(output_daare_path).loudness} LKFS")
    print("What is next?")