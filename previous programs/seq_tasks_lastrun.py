#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Tue Nov  5 17:11:13 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code_5
fb_text  = ''
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'seq_tasks'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"sub_{randint(0, 9999):04.0f}",
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s' % (expName, expInfo['participant'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/strongway/My Drive/_teaching/NCP/Psychophysics/Group2/seq_tasks_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='deg',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'deg'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('space_key') is None:
        # initialise space_key
        space_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='space_key',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Instruction" ---
    text_inst = visual.TextStim(win=win, name='text_inst',
        text="Experimental Instruction\n\nWelcome to our reproduction experiment. In this experiment, you will see a Gabor patch appears on the screen. You need to remember the DURATION of the Gabor patch. After it disappears, a cue (either circle or triangle) will appear. \n\nWhen the cue is CIRCLE, you are asked to press the 'Down' Arrow key as long as what you perceived. The key press will show a Gabor patch again, helping you compare the last duration. \n\nIf the cue is TRIANGLE, a Gabor patch will automatically appear. Your task is to press the 'Down' arrow key as soon as it appears and release the key when it disappears. \n\nPress SPACE to start ...\n\n",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.5, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    space_key = keyboard.Keyboard(deviceName='space_key')
    
    # --- Initialize components for Routine "Encoding" ---
    grating = visual.GratingStim(
        win=win, name='grating',
        tex='sin', mask='gauss', anchor='center',
        ori=1.0, pos=(0, 0), draggable=False, size=(5, 5), sf=1.0, phase=0.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=0.0)
    # Run 'Begin Experiment' code from Code_encoding
    # generate random orientation
    
    orientation = randint(1,180)
    
    
    c_fixation = visual.ShapeStim(
        win=win, name='c_fixation', vertices='cross',
        size=(0.5, 0.5),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "retro_cue" ---
    cue_triangle = visual.ShapeStim(
        win=win, name='cue_triangle',
        size=(0.5, 0.5), vertices='triangle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1, -1.0, 1.0], fillColor=[-1.0000, -1.0000, 1.0000],
        opacity=None, depth=0.0, interpolate=True)
    # Run 'Begin Experiment' code from code
    dur_tri = 0  # cue duration
    dur_cir = 0
    cue_circle = visual.ShapeStim(
        win=win, name='cue_circle',
        size=(0.5, 0.5), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, 0.0039, -1.0000], fillColor=[-1.0000, 0.0039, -1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "Reproduction" ---
    reproduced_grating = visual.GratingStim(
        win=win, name='reproduced_grating',
        tex='sin', mask='gauss', anchor='center',
        ori=1.0, pos=(0, 0), draggable=False, size=(5, 5), sf=1.0, phase=0.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=0.0)
    # Run 'Begin Experiment' code from code_2
    kb = keyboard.Keyboard()
    
    repDuration = 0
    
    # --- Initialize components for Routine "passive" ---
    passive_grating = visual.GratingStim(
        win=win, name='passive_grating',
        tex='sin', mask='gauss', anchor='center',
        ori=1.0, pos=(0, 0), draggable=False, size=(5, 5), sf=1.0, phase=0.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-1.0)
    text_3 = visual.TextStim(win=win, name='text_3',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "feedback" ---
    feedback_text = visual.TextStim(win=win, name='feedback_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.5, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    dummy_1500ms = visual.TextStim(win=win, name='dummy_1500ms',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "BlockInfo" ---
    text_rest = visual.TextStim(win=win, name='text_rest',
        text='Please take a rest! Press any key to continue ...',
        font='Arial',
        pos=(0, 5), draggable=False, height=0.5, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    blk_infotxt = visual.TextStim(win=win, name='blk_infotxt',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.5, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    # Run 'Begin Experiment' code from code_6
    blk_info = ''
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Instruction" ---
    # create an object to store info about Routine Instruction
    Instruction = data.Routine(
        name='Instruction',
        components=[text_inst, space_key],
    )
    Instruction.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for space_key
    space_key.keys = []
    space_key.rt = []
    _space_key_allKeys = []
    # store start times for Instruction
    Instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Instruction.tStart = globalClock.getTime(format='float')
    Instruction.status = STARTED
    Instruction.maxDuration = None
    # keep track of which components have finished
    InstructionComponents = Instruction.components
    for thisComponent in Instruction.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instruction" ---
    Instruction.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_inst* updates
        
        # if text_inst is starting this frame...
        if text_inst.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_inst.frameNStart = frameN  # exact frame index
            text_inst.tStart = t  # local t and not account for scr refresh
            text_inst.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_inst, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_inst.status = STARTED
            text_inst.setAutoDraw(True)
        
        # if text_inst is active this frame...
        if text_inst.status == STARTED:
            # update params
            pass
        
        # *space_key* updates
        waitOnFlip = False
        
        # if space_key is starting this frame...
        if space_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            space_key.frameNStart = frameN  # exact frame index
            space_key.tStart = t  # local t and not account for scr refresh
            space_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(space_key, 'tStartRefresh')  # time at next scr refresh
            # update status
            space_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(space_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(space_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if space_key.status == STARTED and not waitOnFlip:
            theseKeys = space_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _space_key_allKeys.extend(theseKeys)
            if len(_space_key_allKeys):
                space_key.keys = _space_key_allKeys[-1].name  # just the last key pressed
                space_key.rt = _space_key_allKeys[-1].rt
                space_key.duration = _space_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Instruction.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instruction.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instruction" ---
    for thisComponent in Instruction.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Instruction
    Instruction.tStop = globalClock.getTime(format='float')
    Instruction.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "Instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=10.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "Encoding" ---
        # create an object to store info about Routine Encoding
        Encoding = data.Routine(
            name='Encoding',
            components=[grating, c_fixation],
        )
        Encoding.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from Code_encoding
        orientation = randint(1,180)
        # store start times for Encoding
        Encoding.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Encoding.tStart = globalClock.getTime(format='float')
        Encoding.status = STARTED
        thisExp.addData('Encoding.started', Encoding.tStart)
        Encoding.maxDuration = None
        # keep track of which components have finished
        EncodingComponents = Encoding.components
        for thisComponent in Encoding.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Encoding" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        Encoding.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *grating* updates
            
            # if grating is starting this frame...
            if grating.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                grating.frameNStart = frameN  # exact frame index
                grating.tStart = t  # local t and not account for scr refresh
                grating.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(grating, 'tStartRefresh')  # time at next scr refresh
                # update status
                grating.status = STARTED
                grating.setAutoDraw(True)
            
            # if grating is active this frame...
            if grating.status == STARTED:
                # update params
                grating.setOri(orientation, log=False)
            
            # if grating is stopping this frame...
            if grating.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > grating.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    grating.tStop = t  # not accounting for scr refresh
                    grating.tStopRefresh = tThisFlipGlobal  # on global time
                    grating.frameNStop = frameN  # exact frame index
                    # update status
                    grating.status = FINISHED
                    grating.setAutoDraw(False)
            
            # *c_fixation* updates
            
            # if c_fixation is starting this frame...
            if c_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                c_fixation.frameNStart = frameN  # exact frame index
                c_fixation.tStart = t  # local t and not account for scr refresh
                c_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(c_fixation, 'tStartRefresh')  # time at next scr refresh
                # update status
                c_fixation.status = STARTED
                c_fixation.setAutoDraw(True)
            
            # if c_fixation is active this frame...
            if c_fixation.status == STARTED:
                # update params
                pass
            
            # if c_fixation is stopping this frame...
            if c_fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > c_fixation.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    c_fixation.tStop = t  # not accounting for scr refresh
                    c_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    c_fixation.frameNStop = frameN  # exact frame index
                    # update status
                    c_fixation.status = FINISHED
                    c_fixation.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Encoding.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Encoding.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Encoding" ---
        for thisComponent in Encoding.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Encoding
        Encoding.tStop = globalClock.getTime(format='float')
        Encoding.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Encoding.stopped', Encoding.tStop)
        # the Routine "Encoding" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "retro_cue" ---
        # create an object to store info about Routine retro_cue
        retro_cue = data.Routine(
            name='retro_cue',
            components=[cue_triangle, cue_circle],
        )
        retro_cue.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code
        if cue == "reproduction":
            dur_cir = 0.5
            dur_tri = 0
        else:
            dur_tri = 0.5
            dur_cir = 0
        # store start times for retro_cue
        retro_cue.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        retro_cue.tStart = globalClock.getTime(format='float')
        retro_cue.status = STARTED
        retro_cue.maxDuration = None
        # keep track of which components have finished
        retro_cueComponents = retro_cue.components
        for thisComponent in retro_cue.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "retro_cue" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        retro_cue.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cue_triangle* updates
            
            # if cue_triangle is starting this frame...
            if cue_triangle.status == NOT_STARTED and tThisFlip >= 0.3-frameTolerance:
                # keep track of start time/frame for later
                cue_triangle.frameNStart = frameN  # exact frame index
                cue_triangle.tStart = t  # local t and not account for scr refresh
                cue_triangle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_triangle, 'tStartRefresh')  # time at next scr refresh
                # update status
                cue_triangle.status = STARTED
                cue_triangle.setAutoDraw(True)
            
            # if cue_triangle is active this frame...
            if cue_triangle.status == STARTED:
                # update params
                pass
            
            # if cue_triangle is stopping this frame...
            if cue_triangle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_triangle.tStartRefresh + dur_tri-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_triangle.tStop = t  # not accounting for scr refresh
                    cue_triangle.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_triangle.frameNStop = frameN  # exact frame index
                    # update status
                    cue_triangle.status = FINISHED
                    cue_triangle.setAutoDraw(False)
            
            # *cue_circle* updates
            
            # if cue_circle is starting this frame...
            if cue_circle.status == NOT_STARTED and tThisFlip >= 0.3-frameTolerance:
                # keep track of start time/frame for later
                cue_circle.frameNStart = frameN  # exact frame index
                cue_circle.tStart = t  # local t and not account for scr refresh
                cue_circle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_circle, 'tStartRefresh')  # time at next scr refresh
                # update status
                cue_circle.status = STARTED
                cue_circle.setAutoDraw(True)
            
            # if cue_circle is active this frame...
            if cue_circle.status == STARTED:
                # update params
                pass
            
            # if cue_circle is stopping this frame...
            if cue_circle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_circle.tStartRefresh + dur_cir-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_circle.tStop = t  # not accounting for scr refresh
                    cue_circle.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_circle.frameNStop = frameN  # exact frame index
                    # update status
                    cue_circle.status = FINISHED
                    cue_circle.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                retro_cue.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in retro_cue.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "retro_cue" ---
        for thisComponent in retro_cue.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for retro_cue
        retro_cue.tStop = globalClock.getTime(format='float')
        retro_cue.tStopRefresh = tThisFlipGlobal
        # the Routine "retro_cue" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Reproduction" ---
        # create an object to store info about Routine Reproduction
        Reproduction = data.Routine(
            name='Reproduction',
            components=[reproduced_grating],
        )
        Reproduction.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        
        kb.clearEvents(eventType='keyboard')
        event.clearEvents(eventType='keyboard')
        
        key = kb.getKeys(['down'], waitRelease = False, clear = False)
        
        key_pressed = False
        key_released = True
        
        key_onset = 0
        key_offset = 0
        
        repDuration = 0
        # store start times for Reproduction
        Reproduction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Reproduction.tStart = globalClock.getTime(format='float')
        Reproduction.status = STARTED
        thisExp.addData('Reproduction.started', Reproduction.tStart)
        Reproduction.maxDuration = None
        # skip Routine Reproduction if its 'Skip if' condition is True
        Reproduction.skipped = continueRoutine and not (cue == 'following')
        continueRoutine = Reproduction.skipped
        # keep track of which components have finished
        ReproductionComponents = Reproduction.components
        for thisComponent in Reproduction.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Reproduction" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        Reproduction.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *reproduced_grating* updates
            
            # if reproduced_grating is starting this frame...
            if reproduced_grating.status == NOT_STARTED and key_pressed :
                # keep track of start time/frame for later
                reproduced_grating.frameNStart = frameN  # exact frame index
                reproduced_grating.tStart = t  # local t and not account for scr refresh
                reproduced_grating.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reproduced_grating, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reproduced_grating.started')
                # update status
                reproduced_grating.status = STARTED
                reproduced_grating.setAutoDraw(True)
            
            # if reproduced_grating is active this frame...
            if reproduced_grating.status == STARTED:
                # update params
                reproduced_grating.setOri(orientation, log=False)
            
            # if reproduced_grating is stopping this frame...
            if reproduced_grating.status == STARTED:
                if bool(key_released):
                    # keep track of stop time/frame for later
                    reproduced_grating.tStop = t  # not accounting for scr refresh
                    reproduced_grating.tStopRefresh = tThisFlipGlobal  # on global time
                    reproduced_grating.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'reproduced_grating.stopped')
                    # update status
                    reproduced_grating.status = FINISHED
                    reproduced_grating.setAutoDraw(False)
            # Run 'Each Frame' code from code_2
            
            keys = kb.getKeys(['down'], waitRelease=False,clear = False)
            
            for key in keys:
                if key.duration is None:
                    key_pressed = True
                    key_released = False
                    key_onset = key.t
            
            if key_pressed: # after key pressed
                if not keys:
                    key_released = True
                    key_offset =  globalClock.getTime(format='float')
                    continueRoutine = False
                
                
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Reproduction.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Reproduction.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Reproduction" ---
        for thisComponent in Reproduction.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Reproduction
        Reproduction.tStop = globalClock.getTime(format='float')
        Reproduction.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Reproduction.stopped', Reproduction.tStop)
        # Run 'End Routine' code from code_2
        kb.clearEvents(eventType='keyboard')
        event.clearEvents(eventType='keyboard')
        
        #store random orientation in the data
        repDuration = key_offset - key_onset
        thisExp.addData('orientation',orientation)
        thisExp.addData('repDuration', repDuration)
        
        # the Routine "Reproduction" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "passive" ---
        # create an object to store info about Routine passive
        passive = data.Routine(
            name='passive',
            components=[passive_grating, text_3],
        )
        passive.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_4
        
        kb.clearEvents(eventType='keyboard')
        event.clearEvents(eventType='keyboard')
        
        key = kb.getKeys(['down'], waitRelease = False, clear = False)
        
        key_pressed = False
        key_released = True
        
        key_follow_onset = 0
        key_follow_offset = 0
        
        followDuration = 0
        # store start times for passive
        passive.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        passive.tStart = globalClock.getTime(format='float')
        passive.status = STARTED
        passive.maxDuration = None
        # skip Routine passive if its 'Skip if' condition is True
        passive.skipped = continueRoutine and not (cue == 'reproduction')
        continueRoutine = passive.skipped
        # keep track of which components have finished
        passiveComponents = passive.components
        for thisComponent in passive.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "passive" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        passive.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_4
            
            keys = kb.getKeys(['down'], waitRelease=False,clear = False)
            
            for key in keys:
                if key.duration is None:
                    key_pressed = True
                    key_released = False
                    if key_follow_onset == 0:
                        key_follow_onset = key.t
            
            if key_pressed: # after key pressed
                if not keys:
                    key_released = True
                    key_follow_offset = globalClock.getTime(format='float')
                
                
            
            
            # *passive_grating* updates
            
            # if passive_grating is starting this frame...
            if passive_grating.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                passive_grating.frameNStart = frameN  # exact frame index
                passive_grating.tStart = t  # local t and not account for scr refresh
                passive_grating.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(passive_grating, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'passive_grating.started')
                # update status
                passive_grating.status = STARTED
                passive_grating.setAutoDraw(True)
            
            # if passive_grating is active this frame...
            if passive_grating.status == STARTED:
                # update params
                passive_grating.setOri(orientation, log=False)
            
            # if passive_grating is stopping this frame...
            if passive_grating.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > passive_grating.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    passive_grating.tStop = t  # not accounting for scr refresh
                    passive_grating.tStopRefresh = tThisFlipGlobal  # on global time
                    passive_grating.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'passive_grating.stopped')
                    # update status
                    passive_grating.status = FINISHED
                    passive_grating.setAutoDraw(False)
            
            # *text_3* updates
            
            # if text_3 is starting this frame...
            if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_3.frameNStart = frameN  # exact frame index
                text_3.tStart = t  # local t and not account for scr refresh
                text_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_3.status = STARTED
                text_3.setAutoDraw(True)
            
            # if text_3 is active this frame...
            if text_3.status == STARTED:
                # update params
                pass
            
            # if text_3 is stopping this frame...
            if text_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_3.tStartRefresh + 1.0 + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    text_3.tStop = t  # not accounting for scr refresh
                    text_3.tStopRefresh = tThisFlipGlobal  # on global time
                    text_3.frameNStop = frameN  # exact frame index
                    # update status
                    text_3.status = FINISHED
                    text_3.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                passive.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in passive.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "passive" ---
        for thisComponent in passive.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for passive
        passive.tStop = globalClock.getTime(format='float')
        passive.tStopRefresh = tThisFlipGlobal
        # Run 'End Routine' code from code_4
        kb.clearEvents(eventType='keyboard')
        event.clearEvents(eventType='keyboard')
        
        #store random orientation in the data
        followDuration = key_follow_offset - key_follow_onset
        thisExp.addData('passive_key_onset',key_follow_onset)
        thisExp.addData('passive_key_offset', key_follow_offset)
        thisExp.addData('followDuration', followDuration)
        # the Routine "passive" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "feedback" ---
        # create an object to store info about Routine feedback
        feedback = data.Routine(
            name='feedback',
            components=[feedback_text, dummy_1500ms],
        )
        feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_5
        error_ratio = np.min([np.abs(repDuration/duration -1), np.abs(followDuration/duration -1)])
        
        if error_ratio > 0.3:
            fb_text = "Your key pressed duration deviated too far!"
        else:
            fb_text = ''
            
        # store start times for feedback
        feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback.tStart = globalClock.getTime(format='float')
        feedback.status = STARTED
        thisExp.addData('feedback.started', feedback.tStart)
        feedback.maxDuration = None
        # keep track of which components have finished
        feedbackComponents = feedback.components
        for thisComponent in feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_text* updates
            
            # if feedback_text is starting this frame...
            if feedback_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_text.frameNStart = frameN  # exact frame index
                feedback_text.tStart = t  # local t and not account for scr refresh
                feedback_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_text, 'tStartRefresh')  # time at next scr refresh
                # update status
                feedback_text.status = STARTED
                feedback_text.setAutoDraw(True)
            
            # if feedback_text is active this frame...
            if feedback_text.status == STARTED:
                # update params
                feedback_text.setText(fb_text, log=False)
            
            # if feedback_text is stopping this frame...
            if feedback_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_text.tStartRefresh + 0.8-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback_text.tStop = t  # not accounting for scr refresh
                    feedback_text.tStopRefresh = tThisFlipGlobal  # on global time
                    feedback_text.frameNStop = frameN  # exact frame index
                    # update status
                    feedback_text.status = FINISHED
                    feedback_text.setAutoDraw(False)
            
            # *dummy_1500ms* updates
            
            # if dummy_1500ms is starting this frame...
            if dummy_1500ms.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dummy_1500ms.frameNStart = frameN  # exact frame index
                dummy_1500ms.tStart = t  # local t and not account for scr refresh
                dummy_1500ms.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dummy_1500ms, 'tStartRefresh')  # time at next scr refresh
                # update status
                dummy_1500ms.status = STARTED
                dummy_1500ms.setAutoDraw(True)
            
            # if dummy_1500ms is active this frame...
            if dummy_1500ms.status == STARTED:
                # update params
                pass
            
            # if dummy_1500ms is stopping this frame...
            if dummy_1500ms.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dummy_1500ms.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    dummy_1500ms.tStop = t  # not accounting for scr refresh
                    dummy_1500ms.tStopRefresh = tThisFlipGlobal  # on global time
                    dummy_1500ms.frameNStop = frameN  # exact frame index
                    # update status
                    dummy_1500ms.status = FINISHED
                    dummy_1500ms.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback" ---
        for thisComponent in feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback
        feedback.tStop = globalClock.getTime(format='float')
        feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback.stopped', feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if feedback.maxDurationReached:
            routineTimer.addTime(-feedback.maxDuration)
        elif feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        
        # --- Prepare to start Routine "BlockInfo" ---
        # create an object to store info about Routine BlockInfo
        BlockInfo = data.Routine(
            name='BlockInfo',
            components=[text_rest, blk_infotxt, key_resp],
        )
        BlockInfo.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_6
        blknum = trials.thisN // 48
        blk_info = f"That was block #{blknum}. You did great job!"
        print(blk_info)
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # store start times for BlockInfo
        BlockInfo.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        BlockInfo.tStart = globalClock.getTime(format='float')
        BlockInfo.status = STARTED
        thisExp.addData('BlockInfo.started', BlockInfo.tStart)
        BlockInfo.maxDuration = None
        # skip Routine BlockInfo if its 'Skip if' condition is True
        BlockInfo.skipped = continueRoutine and not ((trials.thisN +1) % 48 > 0)
        continueRoutine = BlockInfo.skipped
        # keep track of which components have finished
        BlockInfoComponents = BlockInfo.components
        for thisComponent in BlockInfo.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "BlockInfo" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        BlockInfo.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_rest* updates
            
            # if text_rest is starting this frame...
            if text_rest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_rest.frameNStart = frameN  # exact frame index
                text_rest.tStart = t  # local t and not account for scr refresh
                text_rest.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_rest, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_rest.status = STARTED
                text_rest.setAutoDraw(True)
            
            # if text_rest is active this frame...
            if text_rest.status == STARTED:
                # update params
                pass
            
            # *blk_infotxt* updates
            
            # if blk_infotxt is starting this frame...
            if blk_infotxt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blk_infotxt.frameNStart = frameN  # exact frame index
                blk_infotxt.tStart = t  # local t and not account for scr refresh
                blk_infotxt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blk_infotxt, 'tStartRefresh')  # time at next scr refresh
                # update status
                blk_infotxt.status = STARTED
                blk_infotxt.setAutoDraw(True)
            
            # if blk_infotxt is active this frame...
            if blk_infotxt.status == STARTED:
                # update params
                blk_infotxt.setText(blk_info, log=False)
            
            # if blk_infotxt is stopping this frame...
            if blk_infotxt.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blk_infotxt.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    blk_infotxt.tStop = t  # not accounting for scr refresh
                    blk_infotxt.tStopRefresh = tThisFlipGlobal  # on global time
                    blk_infotxt.frameNStop = frameN  # exact frame index
                    # update status
                    blk_infotxt.status = FINISHED
                    blk_infotxt.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['y','n','left','right','down','space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                BlockInfo.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in BlockInfo.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "BlockInfo" ---
        for thisComponent in BlockInfo.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for BlockInfo
        BlockInfo.tStop = globalClock.getTime(format='float')
        BlockInfo.tStopRefresh = tThisFlipGlobal
        thisExp.addData('BlockInfo.stopped', BlockInfo.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        trials.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            trials.addData('key_resp.rt', key_resp.rt)
            trials.addData('key_resp.duration', key_resp.duration)
        # the Routine "BlockInfo" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 10.0 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if trials.trialList in ([], [None], None):
        params = []
    else:
        params = trials.trialList[0].keys()
    # save data for this loop
    trials.saveAsText(filename + 'trials.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
