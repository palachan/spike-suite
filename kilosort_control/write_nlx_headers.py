# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:18:24 2018

@author: Emily Irvine
"""

def write_ntt_header(nlx_headersize=16*2**10, name=None, t_open=None, t_close=None,
                     filetype=None, fileversion=None, recordsize=None,
                     cheetahrev=None, hardwaresubname=None, hardwaresubtype=None,
                     samplingfreq=None, admaxvalue=None, adbitvolts=None,
                     acqentname=None, numadchannels=None, adchannel=None,
                     inputrange=None, inputinverted=None, amplowcut=None,
                     amphicut=None,ampgain=None, waveformlen=None, alignmentpt=None,
                     threshval=None, minretriggertime=None, spikeretriggertime=None,
                     dualthresh=None, featurepeak1=None, featurepeak2=None,
                     featurepeak3=None, featurepeak4=None, featurevalley1=None,
                     featurevalley2=None, featurevalley3=None, featurevalley4=None,
                     dummy=False):
    """
    Returns a .ntt header
    Parameters
    ----------
    nlx_headersize: float
        Default is 16*2**10
    name: str
        Default is None
    ...
    Returns
    -------
    header: byte string
    """

    if not dummy:
        header = '######## Neuralynx Data File Header' + '\r\n'
    else:
        header = '######## Neuralynx Data File Header - THIS IS A DUMMY HEADER FROM kilosort_control CONTAINING FAKE INFO - SEE ORIGINAL SPIKE FILE FOR RECORDING INFO' + '\r\n'
    if name is not None:
        header += '## File Name ' + name + '\r\n'
    else:
        header += '## File Name ' + '\r\n'
    if t_open is not None:
        header += '## Time Opened (m/d/y): ' + t_open + '\r\n'
    else:
        header += '## Time Opened (m/d/y): ' + '\r\n'
    if t_close is not None:
        header += '## Time Closed (m/d/y): ' + '\r\n'
    else:
        header += '## Time Closed (m/d/y): ' + '\r\n'
    header += '\r\n'
    if filetype is not None:
        header += '-FileType ' + filetype + '\r\n'
    else:
        header += '-FileType ' + '\r\n'
    if fileversion is not None:
        header += '-FileVersion ' + fileversion + '\r\n'
    else:
        header += '-FileVersion ' + '\r\n'
    if recordsize is not None:
        header += '-RecordSize ' + recordsize + '\r\n'
    else:
        header += '-RecordSize ' + '\r\n'
    header += '\r\n'
    if cheetahrev is not None:
        header += '-CheetahRev ' + cheetahrev + '\r\n'
    else:
        header += '-CheetahRev ' + '\r\n'
    header += '\r\n'
    if hardwaresubname is not None:
        header += '-HardwareSubSystemName ' + hardwaresubname + '\r\n'
    else:
        header += '-HardwareSubSystemName ' + '\r\n'
    if hardwaresubtype is not None:
        header += '-HardwareSubSystemType ' + hardwaresubtype + '\r\n'
    else:
        header += '-HardwareSubSystemType ' + '\r\n'
    if samplingfreq is not None:
        header += '-SamplingFrequency ' + samplingfreq + '\r\n'
    else:
        header += '-SamplingFrequency ' + '\r\n'
    if admaxvalue is not None:
        header += '-ADMaxValue ' + admaxvalue + '\r\n'
    else:
        header += '-ADMaxValue ' + '\r\n'
    if adbitvolts is not None:
        header += '-ADBitVolts ' + adbitvolts + '\r\n'
    else:
        header += '-ADBitVolts ' + '\r\n'
    header += '\r\n'
    if acqentname is not None:
        header += '-AcqEntName ' + acqentname + '\r\n'
    else:
        header += '-AcqEntName ' + '\r\n'
    if numadchannels is not None:
        header += '-NumADChannels ' + numadchannels + '\r\n'
    else:
        header += '-NumADChannels ' + '\r\n'
    if adchannel is not None:
        header += '-ADChannel ' + adchannel + '\r\n'
    else:
        header += '-ADChannel ' + '\r\n'
    if inputrange is not None:
        header += '-InputRange ' + inputrange + '\r\n'
    else:
        header += '-InputRange ' + '\r\n'
    if inputinverted is not None:
        header += '-InputInverted ' + inputinverted + '\r\n'
    else:
        header += '-InputInverted ' + '\r\n'
        
    header += '\r\n'
    
    if amplowcut is not None:
        header += '-AmpLowCut ' + amplowcut + '\r\n'
    else:
        header += '-AmpLowCut ' + '\r\n'
    if amphicut is not None:
        header += '-AmpHiCut ' + amphicut + '\r\n'
    else:
        header += '-AmpHiCut ' + '\r\n'
    if ampgain is not None:
        header += '-AmpGain ' + ampgain + '\r\n'
    else:
        header += '-AmpGain ' + '\r\n'
        
    header += '\r\n'
    
    if waveformlen is not None:
        header += '-WaveformLength ' + waveformlen + '\r\n'
    else:
        header += '-WaveformLength ' + '\r\n'
    if alignmentpt is not None:
        header += '-AlignmentPt ' + alignmentpt + '\r\n'
    else:
        header += '-AlignmentPt ' + '\r\n'
    if threshval is not None:
        header += '-ThreshVal ' + threshval + '\r\n'
    else:
        header += '-ThreshVal ' + '\r\n'
    if minretriggertime is not None:
        header += '-MinRetriggerSamples ' + minretriggertime + '\r\n'
    else:
        header += '-MinRetriggerSamples ' + '\r\n'
    if spikeretriggertime is not None:
        header += '-SpikeRetriggerTime ' + spikeretriggertime + '\r\n'
    else:
        header += '-SpikeRetriggerTime ' + '\r\n'
    if dualthresh is not None:
        header += '-DualThresholding ' + dualthresh + '\r\n'
    else:
        header += '-DualThresholding ' + '\r\n'
    header += '\r\n'
    if featurepeak1 is not None:
        header += '-Feature Peak ' + featurepeak1 + '\r\n'
    else:
        header += '-Feature Peak ' + '\r\n'
    if featurepeak2 is not None:
        header += '-Feature Peak ' + featurepeak2 + '\r\n'
    else:
        header += '-Feature Peak ' + '\r\n'
    if featurepeak3 is not None:
        header += '-Feature Peak ' + featurepeak3 + '\r\n'
    else:
        header += '-Feature Peak ' + '\r\n'
    if featurepeak4 is not None:
        header += '-Feature Peak ' + featurepeak4 + '\r\n'
    else:
        header += '-Feature Peak ' + '\r\n'
    if featurevalley1 is not None:
        header += '-Feature Valley ' + featurevalley1 + '\r\n'
    else:
        header += '-Feature Valley ' + '\r\n'
    if featurevalley2 is not None:
        header += '-Feature Valley ' + featurevalley2 + '\r\n'
    else:
        header += '-Feature Valley ' + '\r\n'
    if featurevalley3 is not None:
        header += '-Feature Valley ' + featurevalley3 + '\r\n'
    else:
        header += '-Feature Valley ' + '\r\n'
    if featurevalley4 is not None:
        header += '-Feature Valley ' + featurevalley4 + '\r\n'
    else:
        header += '-Feature Valley ' + '\r\n'
    header += '\r\n'

    offset = int(nlx_headersize - len(header))
    header = header.ljust(offset, '\x00')

    return header.encode()

def write_nst_header(nlx_headersize=16*2**10, name=None, t_open=None, t_close=None,
                     filetype=None, fileversion=None, recordsize=None,
                     cheetahrev=None, hardwaresubname=None, hardwaresubtype=None,
                     samplingfreq=None, admaxvalue=None, adbitvolts=None,
                     acqentname=None, numadchannels=None, adchannel=None,
                     inputrange=None, inputinverted=None, amplowcut=None,
                     amphicut=None,ampgain=None, waveformlen=None, alignmentpt=None,
                     threshval=None, minretriggertime=None, spikeretriggertime=None,
                     dualthresh=None, featurepeak1=None, featurepeak2=None,
                     featurepeak3=None, featurepeak4=None, featurevalley1=None,
                     featurevalley2=None, featurevalley3=None, featurevalley4=None,
                     dummy=False):
    """
    Returns a .nst header
    Parameters
    ----------
    nlx_headersize: float
        Default is 16*2**10
    name: str
        Default is None
    ...
    Returns
    -------
    header: byte string
    """

    if not dummy:
        header = '######## Neuralynx Data File Header' + '\r\n'
    else:
        header = '######## Neuralynx Data File Header - THIS IS A DUMMY HEADER FROM kilosort_control CONTAINING FAKE INFO - SEE ORIGINAL SPIKE FILE FOR RECORDING INFO' + '\r\n'

    if name is not None:
        header += '## File Name ' + name + '\r\n'
    else:
        header += '## File Name ' + '\r\n'
    if t_open is not None:
        header += '## Time Opened (m/d/y): ' + t_open + '\r\n'
    else:
        header += '## Time Opened (m/d/y): ' + '\r\n'
    if t_close is not None:
        header += '## Time Closed (m/d/y): ' + '\r\n'
    else:
        header += '## Time Closed (m/d/y): ' + '\r\n'
    header += '\r\n'
    if filetype is not None:
        header += '-FileType ' + filetype + '\r\n'
    else:
        header += '-FileType ' + '\r\n'
    if fileversion is not None:
        header += '-FileVersion ' + fileversion + '\r\n'
    else:
        header += '-FileVersion ' + '\r\n'
    if recordsize is not None:
        header += '-RecordSize ' + recordsize + '\r\n'
    else:
        header += '-RecordSize ' + '\r\n'
    header += '\r\n'
    if cheetahrev is not None:
        header += '-CheetahRev ' + cheetahrev + '\r\n'
    else:
        header += '-CheetahRev ' + '\r\n'
    header += '\r\n'
    if hardwaresubname is not None:
        header += '-HardwareSubSystemName ' + hardwaresubname + '\r\n'
    else:
        header += '-HardwareSubSystemName ' + '\r\n'
    if hardwaresubtype is not None:
        header += '-HardwareSubSystemType ' + hardwaresubtype + '\r\n'
    else:
        header += '-HardwareSubSystemType ' + '\r\n'
    if samplingfreq is not None:
        header += '-SamplingFrequency ' + samplingfreq + '\r\n'
    else:
        header += '-SamplingFrequency ' + '\r\n'
    if admaxvalue is not None:
        header += '-ADMaxValue ' + admaxvalue + '\r\n'
    else:
        header += '-ADMaxValue ' + '\r\n'
    if adbitvolts is not None:
        header += '-ADBitVolts ' + adbitvolts + '\r\n'
    else:
        header += '-ADBitVolts ' + '\r\n'
    header += '\r\n'
    if acqentname is not None:
        header += '-AcqEntName ' + acqentname + '\r\n'
    else:
        header += '-AcqEntName ' + '\r\n'
    if numadchannels is not None:
        header += '-NumADChannels ' + numadchannels + '\r\n'
    else:
        header += '-NumADChannels ' + '\r\n'
    if adchannel is not None:
        header += '-ADChannel ' + adchannel + '\r\n'
    else:
        header += '-ADChannel ' + '\r\n'
    if inputrange is not None:
        header += '-InputRange ' + inputrange + '\r\n'
    else:
        header += '-InputRange ' + '\r\n'
    if inputinverted is not None:
        header += '-InputInverted ' + inputinverted + '\r\n'
    else:
        header += '-InputInverted ' + '\r\n'
        
    header += '\r\n'
    
    if amplowcut is not None:
        header += '-AmpLowCut ' + amplowcut + '\r\n'
    else:
        header += '-AmpLowCut ' + '\r\n'
    if amphicut is not None:
        header += '-AmpHiCut ' + amphicut + '\r\n'
    else:
        header += '-AmpHiCut ' + '\r\n'
    if ampgain is not None:
        header += '-AmpGain ' + ampgain + '\r\n'
    else:
        header += '-AmpGain ' + '\r\n'
        
    header += '\r\n'
    
    if waveformlen is not None:
        header += '-WaveformLength ' + waveformlen + '\r\n'
    else:
        header += '-WaveformLength ' + '\r\n'
    if alignmentpt is not None:
        header += '-AlignmentPt ' + alignmentpt + '\r\n'
    else:
        header += '-AlignmentPt ' + '\r\n'
    if threshval is not None:
        header += '-ThreshVal ' + threshval + '\r\n'
    else:
        header += '-ThreshVal ' + '\r\n'
    if minretriggertime is not None:
        header += '-MinRetriggerSamples ' + minretriggertime + '\r\n'
    else:
        header += '-MinRetriggerSamples ' + '\r\n'
    if spikeretriggertime is not None:
        header += '-SpikeRetriggerTime ' + spikeretriggertime + '\r\n'
    else:
        header += '-SpikeRetriggerTime ' + '\r\n'
    if dualthresh is not None:
        header += '-DualThresholding ' + dualthresh + '\r\n'
    else:
        header += '-DualThresholding ' + '\r\n'
    header += '\r\n'
    if featurepeak1 is not None:
        header += '-Feature Peak ' + featurepeak1 + '\r\n'
    else:
        header += '-Feature Peak 0 0 ' + '\r\n'
    if featurepeak2 is not None:
        header += '-Feature Peak ' + featurepeak2 + '\r\n'
    else:
        header += '-Feature Peak 1 1 ' + '\r\n'
    if featurepeak3 is not None:
        header += '-Feature Peak ' + featurepeak3 + '\r\n'
    else:
        header += '-Feature Valley 2 0 ' + '\r\n'
    if featurepeak4 is not None:
        header += '-Feature Peak ' + featurepeak4 + '\r\n'
    else:
        header += '-Feature Valley 3 1 ' + '\r\n'
    if featurevalley1 is not None:
        header += '-Feature Valley ' + featurevalley1 + '\r\n'
    else:
        header += '-Feature Energy 4 0 ' + '\r\n'
    if featurevalley2 is not None:
        header += '-Feature Valley ' + featurevalley2 + '\r\n'
    else:
        header += '-Feature Energy 5 1 ' + '\r\n'
    if featurevalley3 is not None:
        header += '-Feature Valley ' + featurevalley3 + '\r\n'
    else:
        header += '-Feature Height 6 0 ' + '\r\n'
    if featurevalley4 is not None:
        header += '-Feature Valley ' + featurevalley4 + '\r\n'
    else:
        header += '-Feature Height 7 1 ' + '\r\n'
    header += '\r\n'

    offset = int(nlx_headersize - len(header))
    header = header.ljust(offset, '\x00')

    return header.encode()